import numpy as np

from .controller_base import ControllerBase
from ...core.types import RobotCmd, RobotState, LocalMotionPlan, ControlMode


class JointPDController(ControllerBase):
    """Tracks joint position and velocity reference provided in `local_plan` message directly."""

    def __init__(
        self,
        kp: np.ndarray | float,
        kd: np.ndarray | float,
        hold_position_at_start: bool = True,
        gain_ramp_up_time: float = 0.0,
        ramp_start_kp: np.ndarray | float = 0.0,
        ramp_start_kd: np.ndarray | float = 0.0,
    ):
        """Tracks joint position and velocity reference using effort commands (PD law).

        When `local_plan.control_mode` is `CONTROL`, this controller will create a command to
        track `joint_positions` and `joint_velocities` values from `local_plan`.

        Args:
            kp (np.ndarray | float): Joint position/stiffness gains.
            kd (np.ndarray | float): Joint velocity/damping gains.
            hold_position_at_start (bool, optional): If set to True, will hold the last joint
                position when control_mode is ControlMode.IDLE. Otherwise, will track current
                joint state positions of the robot (can drift). Defaults to True.
            gain_ramp_up_time (float, optional): Optionally, the controller can exponentially
                ramp up the gains to the desired stiffness and damping gains. Defaults to 0.0.
            ramp_start_kp (np.ndarray | float, optional): Starting value for stiffness gains
                when `gain_ramp_up_time` > 0.0. Defaults to 0.0.
            ramp_start_kd (np.ndarray | float, optional): Starting value for damping gains
                when `gain_ramp_up_time` > 0.0. Defaults to 0.0.
        """
        self._target_kp = np.array(kp).flatten()
        self._target_kd = np.array(kd).flatten()
        self._hold_position = hold_position_at_start

        # gain ramping stuff
        self._ramp_duration = gain_ramp_up_time if gain_ramp_up_time > 0.0 else -1
        self._ramp_start_kp = np.array(ramp_start_kp).flatten()
        self._ramp_start_kd = np.array(ramp_start_kd).flatten()
        self._on_ramp_target = (
            np.log(self._ramp_duration + 1) / self._ramp_duration
            if self._ramp_duration > 0
            else 0.0
        )
        self._ramp_factor = 1.0  # expose ramp factor so derived controllers can use it
        self._duration_in_ctrl_mode = 0.0
        self._ramping_complete = False

        self._robot_cmd: RobotCmd = None

    def update(
        self,
        robot_state: RobotState,
        local_plan: LocalMotionPlan,
        t: float = None,
        dt: float = None,
    ) -> RobotCmd:
        if self._robot_cmd is None:
            self._robot_cmd: RobotCmd = RobotCmd.createZeros(
                dof=len(robot_state.joint_states.joint_names),
                joint_names=robot_state.joint_states.joint_names,
            )
            self._robot_cmd.Kp = self._ramp_factor * self._target_kp
            self._robot_cmd.Kd = self._ramp_factor * self._target_kd
            self._robot_cmd.joint_commands.joint_positions = (
                robot_state.joint_states.joint_positions.copy()
            )
        if local_plan is not None:
            match local_plan.control_mode:
                case ControlMode.IDLE:
                    if not self._hold_position:
                        # NOTE: this will not hold the robot in place, but will still try to
                        # track the current joint positions (which may result in drift due to
                        # tracking errors). To hold in place, we have to record first state
                        # and continuously track that.
                        self._robot_cmd.joint_commands.joint_positions = (
                            robot_state.joint_states.joint_positions
                        )
                        self._robot_cmd.joint_commands.joint_names = (
                            robot_state.joint_states.joint_names
                        )
                        self._robot_cmd.joint_commands.joint_velocities = np.zeros_like(
                            robot_state.joint_states.joint_velocities
                        )
                case ControlMode.CONTROL:
                    if local_plan.joint_references.joint_names is None:
                        return self._robot_cmd

                    self._robot_cmd.joint_commands.joint_positions = (
                        local_plan.joint_references.joint_positions
                    )
                    self._robot_cmd.joint_commands.joint_names = (
                        local_plan.joint_references.joint_names
                    )
                    self._robot_cmd.joint_commands.joint_velocities = (
                        local_plan.joint_references.joint_velocities
                    )
                case _:
                    raise RuntimeError(
                        f"Invalid control_mode in local plan message: {local_plan.control_mode}."
                    )

        if self._duration_in_ctrl_mode <= self._ramp_duration:
            self._ramping_complete = False
            self._ramp_factor = (
                np.exp(self._on_ramp_target * self._duration_in_ctrl_mode) - 1
            ) / self._ramp_duration
            self._robot_cmd.Kp = self._ramp_start_kp + self._ramp_factor * (
                self._target_kp - self._ramp_start_kp
            )
            self._robot_cmd.Kd = self._ramp_start_kd + self._ramp_factor * (
                self._target_kd - self._ramp_start_kd
            )
        elif not self._ramping_complete:
            self._ramp_factor = 1.0
            self._robot_cmd.Kp = self._target_kp
            self._robot_cmd.Kd = self._target_kd
            self._ramping_complete = True
        self._duration_in_ctrl_mode += dt

        return self._robot_cmd
