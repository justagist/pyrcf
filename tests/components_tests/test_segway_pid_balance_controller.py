"""Tests for the takeoff (loss of ground contact) handling of `SegwayPIDBalanceController`.

The controller's air-return branch was previously unreachable (the contact check always evaluated
truthy) and, once reached, called `low_pass_filter` with swapped arguments. These tests pin down
both behaviours.
"""

import numpy as np
import pytest

from pyrcf.components.controllers import SegwayPIDBalanceController
from pyrcf.core.types import (
    ControlMode,
    EndEffectorStates,
    JointStates,
    LocalMotionPlan,
    Pose3D,
    RobotState,
    StateEstimates,
    Twist,
)


class FakePinocchioInterface:
    """Minimal stand-in: the controller only queries end-effector poses from pinocchio."""

    def __init__(self, wheel_x: float = 0.0):
        self.wheel_x = wheel_x

    def get_ee_poses(self):
        pos = np.array([self.wheel_x, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        return [(pos, quat), (pos, quat)]


def make_robot_state(contact_states, upright: bool = True):
    # pitched-over quaternion is 1.2 rad about y (default `fall_pitch` is 1.0 rad); deliberately
    # not exactly pi/2, which would sit on gimbal lock in the quat->rpy conversion
    orientation = (
        np.array([0.0, 0.0, 0.0, 1.0])
        if upright
        else np.array([0.0, np.sin(0.6), 0.0, np.cos(0.6)])
    )
    return RobotState(
        joint_states=JointStates(
            joint_names=["j0", "j1", "left_wheel", "j3", "j4", "right_wheel"],
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_efforts=np.zeros(6),
        ),
        state_estimates=StateEstimates(
            pose=Pose3D(position=np.zeros(3), orientation=orientation),
            twist=Twist(),
            end_effector_states=EndEffectorStates(
                ee_names=["left_wheel_tire", "right_wheel_tire"],
                contact_states=contact_states,
            ),
        ),
    )


@pytest.fixture(name="controller")
def controller_fixture():
    return SegwayPIDBalanceController(
        pinocchio_interface=FakePinocchioInterface(wheel_x=0.3),
        air_return_period=1.0,
    )


def idle_plan():
    return LocalMotionPlan(twist=Twist(), control_mode=ControlMode.CONTROL)


class TestTakeoffDetection:

    def test_no_contact_is_detected(self, controller: SegwayPIDBalanceController):
        """With both wheels off the ground the air-return branch must run without raising.

        This used to raise AssertionError from `low_pass_filter` because dt and cutoff_period
        were passed in the wrong positions.
        """
        controller.integral_error_velocity = 0.5
        controller.target_ground_position = 0.0

        controller.compute_wheel_velocities(
            robot_state=make_robot_state(contact_states=[0, 0]),
            local_plan=idle_plan(),
            dt=0.005,
        )

        # integral error must decay towards zero while airborne
        assert 0.0 < controller.integral_error_velocity < 0.5
        # ground position target must move towards the measured ground position (0.3)
        assert 0.0 < controller.target_ground_position < 0.3

    def test_integral_error_decays_to_zero_while_airborne(
        self, controller: SegwayPIDBalanceController
    ):
        controller.integral_error_velocity = 1.0
        robot_state = make_robot_state(contact_states=[0, 0])

        for _ in range(2000):
            controller.compute_wheel_velocities(robot_state, idle_plan(), dt=0.005)

        assert controller.integral_error_velocity == pytest.approx(0.0, abs=1e-4)

    def test_contact_on_either_wheel_counts_as_grounded(
        self, controller: SegwayPIDBalanceController
    ):
        controller.integral_error_velocity = 0.0
        # one wheel down: the grounded branch integrates the error rather than decaying it
        controller.compute_wheel_velocities(
            make_robot_state(contact_states=[0, 1]), idle_plan(), dt=0.005
        )
        assert controller.integral_error_velocity != 0.0

    def test_missing_contact_states_assumes_grounded(self, controller: SegwayPIDBalanceController):
        controller.integral_error_velocity = 0.0
        controller.compute_wheel_velocities(
            make_robot_state(contact_states=None), idle_plan(), dt=0.005
        )
        assert controller.integral_error_velocity != 0.0

    def test_fall_is_detected(self, controller: SegwayPIDBalanceController):
        with pytest.raises(RuntimeError, match="denotes a fall"):
            controller.compute_wheel_velocities(
                make_robot_state(contact_states=[1, 1], upright=False),
                idle_plan(),
                dt=0.005,
            )
