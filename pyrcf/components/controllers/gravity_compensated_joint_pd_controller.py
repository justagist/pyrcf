import numpy as np

from .joint_pd_controller import JointPDController
from ...core.types import RobotCmd, RobotState, LocalMotionPlan
from ...utils.kinematics_dynamics.pinocchio_interface import PinocchioInterface

# pylint: disable=W0212


class GravityCompensatedPDController(JointPDController):
    """An extension of the `JointPDController` that adds gravity compensation
    torques to the commands."""

    def __init__(
        self,
        kp: np.ndarray | float,
        kd: np.ndarray | float,
        pinocchio_interface: PinocchioInterface,
        hold_position_at_start: bool = True,
        gain_ramp_up_time: float = 0.0,
        ramp_start_kp: np.ndarray | float = 0.0,
        ramp_start_kd: np.ndarray | float = 0.0,
        ramp_gravity_torques: bool = True,
    ):
        """An extension of the `JointPDController` that adds gravity compensation
        torques to the commands.

        When `local_plan.control_mode` is `IDLE`, this controller will simply try to track the
        current joint state of the robot.
            NOTE: this will not necessarily hold the robot in place, it will try to
            track the current joint positions (which may result in drift due to
            tracking errors.) To hold in place, will have to record first state
            and continuously track that. (See `hold_position_controller.py`.)
        When `local_plan.control_mode` is `CONTROL`, this controller will create a command to
        track `joint_positions` and `joint_velocities` values from `local_plan`.

        Args:
            kp (np.ndarray | float): Joint position/stiffness gains.
            kd (np.ndarray | float): Joint velocity/damping gains.
            pinocchio_interface (PinocchioInterface): The PinocchioInterface object
                associated with the robot. This object should be externally updated, such
                as when using the `BulletRobot` class's `read()` method.
            gain_ramp_up_time (float, optional): Optionally, the controller can exponentially
                ramp up the gains to the desired stiffness and damping gains. Defaults to 0.0.
                NOTE: This ramping up also happens to the gravity compensation torques by
                default. To disable this, set `ramp_gravity_torques=False`.
            ramp_start_kp (np.ndarray | float, optional): Starting value for stiffness gains
                when `gain_ramp_up_time` > 0.0. Defaults to 0.0.
            ramp_start_kd (np.ndarray | float, optional): Starting value for damping gains
                when `gain_ramp_up_time` > 0.0. Defaults to 0.0.
        """
        super().__init__(
            kp=kp,
            kd=kd,
            hold_position_at_start=hold_position_at_start,
            gain_ramp_up_time=gain_ramp_up_time,
            ramp_start_kp=ramp_start_kp,
            ramp_start_kd=ramp_start_kd,
        )

        self._ramp_gravity = ramp_gravity_torques if gain_ramp_up_time > 0.0 else False

        self._pin_iface = pinocchio_interface
        self._compensate = True

        self._j_id_offset = 6 if self._pin_iface.floating_base else 0

    def toggle_compensation(self, enable: bool = None):
        """Enable or disable gravity compensation by applying additional torque
        commands to compensate robot's static weight.

        Args:
            enable (bool, optional): True to enable, False to disable. Defaults to None,
                (when None provided, will automatically toggle between enable/disable).
        """
        if enable is None:
            self._compensate = not self._compensate
        else:
            self._compensate = bool(enable)

    def update(
        self,
        robot_state: RobotState,
        local_plan: LocalMotionPlan,
        t: float = None,
        dt: float = None,
    ) -> RobotCmd:
        super().update(robot_state, local_plan, t, dt)

        # compute gravity torques using pinocchio and add this as the feedforward
        # torque in the robot command.
        if self._compensate:
            g = self._pin_iface.get_gravity_vector()
            for pin_j_id, jname in enumerate(self._pin_iface.actuated_joint_names):
                self._robot_cmd.joint_commands.joint_efforts[
                    self._robot_cmd.joint_commands._joint_name_to_idx[jname]
                ] = (self._ramp_factor if self._ramp_gravity else 1) * g[
                    self._j_id_offset + pin_j_id
                ]
        else:
            self._robot_cmd.joint_commands.joint_efforts *= 0.0

        return self._robot_cmd
