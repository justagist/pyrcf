"""
Adapted from wheel_controller from Stéphane Caron's upkie repo.
(https://github.com/upkie/upkie/blob/main/pid_balancer/wheel_controller.py).

Most of the comments from the original code still included.

# License from upkie repo code included below:

# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from .controller_base import ControllerBase
from ...core.types import ControlMode, LocalMotionPlan, RobotCmd, RobotState
from ...utils.math_utils import quat2rpy
from ...utils.filters import abs_bounded_derivative_filter, low_pass_filter
from ...utils.kinematics_dynamics.pinocchio_interface import PinocchioInterface


class SegwayPIDBalanceController(ControllerBase):
    """A simple PD balance controller for a 2-wheel segway type robot."""

    @dataclass
    class Gains:
        """
        Gains for this controller.

        Args:
            pitch_damping: Pitch error (normalized) damping gain.
                Corresponds to the proportional term of the velocity PI
                controller, equivalent to the derivative term of the
                acceleration PD controller.
            pitch_stiffness: Pitch error (normalized) stiffness gain.
                Corresponds to the integral term of the velocity PI
                controller, equivalent to the proportional term of the
                acceleration PD controller.
            footprint_position_damping: Position error (normalized) damping gain.
                Corresponds to the proportional term of the velocity PI
                controller, equivalent to the derivative term of the
                acceleration PD controller.
            footprint_position_stiffness: Position error (normalized) stiffness gain.
                Corresponds to the integral term of the velocity PI
                controller, equivalent to the proportional term of the
                acceleration PD controller.
            joint_kp: Joint position tracking gain (for non-wheel joints).
            joint_kd: Wheel joint velocity gain (only for wheel joints)
            turning_gain_scale: joint velocity additional gains when there
                is turning probability (yaw command).
        """

        pitch_damping: float = 10.0
        pitch_stiffness: float = 20.0
        footprint_position_damping: float = 5.0
        footprint_position_stiffness: float = 2.0
        joint_kp: float = 200.0
        joint_kd: float = 2.0
        turning_gain_scale: float = 2.0

    def __init__(
        self,
        pinocchio_interface: PinocchioInterface,
        gains: Gains = Gains(),
        air_return_period: float = 1.0,
        fall_pitch: float = 1.0,
        max_ground_velocity: float = 2.0,
        max_integral_error_velocity: float = 10.0,
        max_target_accel: float = 0.3,
        max_target_distance: float = 1.5,
        max_target_velocity: float = 0.6,
        max_yaw_accel: float = 10.0,
        max_yaw_velocity: float = 1.0,
        turning_deadband: float = 0.015,
        turning_decision_time: float = 0.2,
        wheel_radius: float = 0.06,
        wheel_distance: float = 0.3048,
        wheel_ee_ids: List[int] = [0, 1],
        wheel_joint_ids: List[int] = [2, 5],
        wheel_joint_directions: List[int] = [1, -1],
        joint_torque_limits: List[float] = [-1.0, 1.0],
    ):
        """A simple PD balance controller for a 2-wheel segway type robot.

        Initialize balancer.

        Args:
            air_return_period: Cutoff period for resetting integrators while
                the robot is in the air, in [s].
            max_ground_velocity: Maximum commanded ground velocity no matter
                what, in [m] / [s].
            max_integral_error_velocity: Maximum integral error velocity, in
                [m] / [s].
            max_target_accel: Maximum acceleration for the ground
                target, in [m] / [s]². This bound does not affect the commanded
                ground velocity.
            max_target_distance: Maximum distance from the current ground
                position to the target, in [m].
            max_target_velocity: Maximum velocity for the ground target,
                in [m] / [s]. This bound indirectly affects the commanded
                ground velocity.
            max_yaw_accel: Maximum yaw angular acceleration in [rad] / [s]².
            max_yaw_velocity: Maximum yaw angular velocity in [rad] / [s].
            turning_deadband: Joystick axis value between 0.0 and 1.0 below
                which legs stiffen but the turning motion doesn't start.
            turning_decision_time: Minimum duration in [s] for the turning
                probability to switch from zero to one and converesly.
            wheel_radius: Wheel radius in [m].
        """
        assert 0.0 <= turning_deadband <= 1.0
        self.air_return_period = air_return_period
        self.fall_pitch = fall_pitch
        self.gains = gains
        self.integral_error_velocity = 0.0
        self.max_ground_velocity = max_ground_velocity
        self.max_integral_error_velocity = max_integral_error_velocity
        self.max_target_accel = max_target_accel
        self.max_target_distance = max_target_distance
        self.max_target_velocity = max_target_velocity
        self.max_yaw_accel = max_yaw_accel
        self.max_yaw_velocity = max_yaw_velocity
        self.pitch = 0.0
        self.target_ground_position = 0.0
        self.target_ground_velocity = 0.0
        self.target_yaw_position = 0.0
        self.target_yaw_velocity = 0.0
        self.turning_deadband = turning_deadband
        self.turning_decision_time = turning_decision_time
        self.turning_probability = 0.0
        self.wheel_radius = wheel_radius
        self.wheel_ee_ids = wheel_ee_ids
        self.wheel_distance = wheel_distance
        self.wheel_joint_ids = wheel_joint_ids
        self.torque_lims = joint_torque_limits
        self.wheel_joint_directions = wheel_joint_directions

        self._pin = pinocchio_interface

        self.kp_ground_vel = np.array(
            [
                gains.footprint_position_damping,
                gains.pitch_damping,
            ]
        )
        self.ki_ground_vel = np.array(
            [
                gains.footprint_position_stiffness,
                gains.pitch_stiffness,
            ]
        )

        self._reference_joint_pos = None
        self._cmd: RobotCmd = None

    def compute_wheel_velocities(
        self,
        robot_state: RobotState,
        local_plan: LocalMotionPlan,
        dt: float = None,
    ) -> np.ndarray:
        self.target_ground_velocity = abs_bounded_derivative_filter(
            self.target_ground_velocity,
            local_plan.twist.linear[0],
            dt,
            self.max_target_velocity,
            self.max_target_accel,
        )

        turning_intent = abs(local_plan.twist.angular[2] / self.turning_deadband)
        self.turning_probability = abs_bounded_derivative_filter(
            self.turning_probability,
            turning_intent,  # might be > 1.0
            dt,
            max_output=1.0,  # output is <= 1.0
            max_derivative=1.0 / self.turning_decision_time,
        )

        velocity_ratio = (abs(local_plan.twist.angular[2]) - self.turning_deadband) / (
            1.0 - self.turning_deadband
        )
        velocity_ratio = max(0.0, velocity_ratio)
        yaw_vel = self.max_yaw_velocity * np.sign(local_plan.twist.angular[2]) * velocity_ratio
        turn_hasnt_started = abs(self.target_yaw_velocity) < 0.01
        turn_not_sure_yet = self.turning_probability < 0.99
        if turn_hasnt_started and turn_not_sure_yet:
            yaw_vel = 0.0
        self.target_yaw_velocity = abs_bounded_derivative_filter(
            self.target_yaw_velocity,
            yaw_vel,
            dt,
            self.max_yaw_velocity,
            self.max_yaw_accel,
        )
        if abs(self.target_yaw_velocity) > 0.01:  # still turning
            self.turning_probability = 1.0

        torso_pitch = quat2rpy(robot_state.state_estimates.pose.orientation)[1]
        if abs(torso_pitch) > self.fall_pitch:
            self.integral_error_velocity = 0.0  # [m] / [s]
            ground_velocity = 0.0  # [m] / [s]
            raise RuntimeError(f"Base angle {torso_pitch=:.3} rad denotes a fall")

        # NOTE: this is a weird way of setting ee position reference. It is not transformed to base frame!!
        # NOTE: also this is averaging over all end-effector poses (will break if
        # the robot has end-effectors other than wheels)
        ground_position = np.mean([ee_pose[0][0] for ee_pose in self._pin.get_ee_poses()])
        floor_contact = np.any(
            robot_state.state_estimates.contact_states[i] for i in self.wheel_ee_ids
        )

        error = np.array(
            [
                self.target_ground_position - ground_position,
                0.0 - torso_pitch,  # target pitch is 0.0 rad
            ]
        )

        if not floor_contact:
            self.integral_error_velocity = low_pass_filter(
                self.integral_error_velocity, self.air_return_period, 0.0, dt
            )
            # We don't reset self.target_ground_velocity: either takeoff
            # detection is a false positive and we should resume close to the
            # pre-takeoff state, or the robot is really in the air and the user
            # should stop smashing the joystick like a bittern ;p
            self.target_ground_position = low_pass_filter(
                self.target_ground_position,
                self.air_return_period,
                ground_position,
                dt,
            )
        else:  # floor_contact:
            self.integral_error_velocity += self.ki_ground_vel.dot(error) * dt
            self.integral_error_velocity = np.clip(
                self.integral_error_velocity,
                -self.max_integral_error_velocity,
                self.max_integral_error_velocity,
            )
            self.target_ground_position += self.target_ground_velocity * dt
            self.target_ground_position = np.clip(
                self.target_ground_position,
                ground_position - self.max_target_distance,
                ground_position + self.max_target_distance,
            )

        # Non-minimum phase trick: as per control theory's book, the proper
        # feedforward velocity should be ``+self.target_ground_velocity``.
        # However, it is with resolute purpose that it sends
        # ``-self.target_ground_velocity`` instead!
        #
        # Try both on the robot, you will see the difference :)
        #
        # This hack is not purely out of "esprit de contradiction". Changing
        # velocity is a non-minimum phase behavior (to accelerate forward, the
        # ZMP of the LIPM needs to move backward at first, then forward), and
        # our feedback can't realize that (it only takes care of balancing
        # around a stationary velocity).
        #
        # What's left? Our integrator! If we send the opposite of the target
        # velocity (or only a fraction of it, although 100% seems to do a good
        # job), Upkie will immediately start executing the desired non-minimum
        # phase behavior. The error will then grow and the integrator catch up
        # so that ``upkie_trick_velocity - self.integral_error_velocity``
        # converges to its proper steady state value (the same value ``0 -
        # self.integral_error_velocity`` would have converged to if we had no
        # feedforward).
        #
        # Unconvinced? Try it on the robot. You will feel Upkie's trick ;)
        #
        upkie_trick_velocity = -self.target_ground_velocity

        ground_velocity = (
            upkie_trick_velocity - self.kp_ground_vel.dot(error) - self.integral_error_velocity
        )
        ground_velocity = np.clip(
            ground_velocity, -self.max_ground_velocity, self.max_ground_velocity
        )

        left_wheel_velocity = self.wheel_joint_directions[0] * ground_velocity / self.wheel_radius
        right_wheel_velocity = self.wheel_joint_directions[1] * ground_velocity / self.wheel_radius

        # Yaw rotation
        turning_radius = 0.5 * self.wheel_distance
        yaw_to_wheel = turning_radius / self.wheel_radius

        # NOTE: not sure why the target yaw velocities have to be negative (not there in original upkie code)
        left_wheel_velocity += yaw_to_wheel * (-self.target_yaw_velocity)
        right_wheel_velocity += yaw_to_wheel * (-self.target_yaw_velocity)

        return np.array([left_wheel_velocity, right_wheel_velocity])

    def update(  # pylint: disable=W0222
        self,
        robot_state: RobotState,
        local_plan: LocalMotionPlan,
        t: float,
        dt: float,
    ) -> RobotCmd:
        if self._reference_joint_pos is None:
            self._reference_joint_pos = robot_state.joint_states.joint_positions

        if local_plan.control_mode != ControlMode.CONTROL:
            return RobotCmd()

        target_vel = np.zeros(robot_state.joint_states.joint_positions.size)
        target_vel[self.wheel_joint_ids] = self.compute_wheel_velocities(
            robot_state=robot_state, local_plan=local_plan, dt=dt
        )
        self._reference_joint_pos[self.wheel_joint_ids] = robot_state.joint_states.joint_positions[
            self.wheel_joint_ids
        ]

        kd = self.gains.joint_kd + self.gains.turning_gain_scale * self.turning_probability

        if self._cmd is None:
            self._cmd = RobotCmd.createZeros(
                dof=robot_state.joint_states.joint_positions.size,
                joint_names=robot_state.joint_states.joint_names,
            )

        self._cmd.joint_commands.joint_efforts = self.gains.joint_kp * (
            self._reference_joint_pos - robot_state.joint_states.joint_positions
        )

        self._cmd.joint_commands.joint_efforts += kd * (
            target_vel - robot_state.joint_states.joint_velocities
        )

        self._cmd.joint_commands.joint_efforts = np.clip(
            self._cmd.joint_commands.joint_efforts,
            self.torque_lims[0],
            self.torque_lims[1],
        )

        return self._cmd
