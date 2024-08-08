import copy
import numpy as np

from .blind_forwarding_planner import LocalPlannerBase
from ...utils.filters import second_order_filter
from ...core.types import GlobalMotionPlan, RobotState, LocalMotionPlan, ControlMode, PlannerMode


class JointReferenceInterpolator(LocalPlannerBase):
    """Simply interpolates joint references given through the global plan messages using
    a 2nd-order filter."""

    def __init__(
        self,
        filter_gain=0.05,
        blind_mode: bool = True,
        forward_other_global_plan_values: bool = True,
    ):
        """A "local planner" that simply interpolates joint references given
        through the global plan messages using a 2nd-order filter.

        Args:
            filter_gain (float, optional): Smoothing gain for second-order filter
                interpolator. Defaults to 0.05.
            blind_mode (bool, optional): If set to False, will not use joint state
                feedback for generating reference (will continue from previous target).
                Defaults to True.
            forward_other_global_plan_values (bool, optional): If set to True, all
                other values from the global plan message (such as twist, relative_pose,
                end_effector_references) are copied to the output local_plan message from
                this planner.
        """
        self._alpha = filter_gain
        self._blind_mode = blind_mode
        self._output_plan: LocalMotionPlan = None
        self._prev_ref: np.ndarray = None
        self._forward_gp = forward_other_global_plan_values

    def generate_local_plan(
        self,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        t: float,
        dt: float,
    ) -> LocalMotionPlan:

        if global_plan.planner_mode != PlannerMode.CUSTOM:
            self._output_plan = None
            return LocalMotionPlan()

        if global_plan.joint_references.joint_positions is not None:
            if self._output_plan is None:
                self._output_plan = LocalMotionPlan(control_mode=ControlMode.CONTROL)
                self._output_plan.joint_references.joint_names = copy.deepcopy(
                    global_plan.joint_references.joint_names
                )
                self._prev_ref = np.array(
                    [
                        robot_state.joint_states.get_state_of(jname)[0]
                        for jname in global_plan.joint_references.joint_names
                    ]
                )
            if self._blind_mode:
                ref_joints = self._prev_ref
            else:
                ref_joints = np.array(
                    [
                        robot_state.joint_states.get_state_of(jname)[0]
                        for jname in global_plan.joint_references.joint_names
                    ]
                )
            self._output_plan.joint_references.joint_positions = second_order_filter(
                current_value=ref_joints,
                desired_value=global_plan.joint_references.joint_positions,
                gain=self._alpha,
            )
            if dt <= 0.0:
                self._output_plan.joint_references.joint_velocities = (
                    self._output_plan.joint_references.joint_positions * 0
                )
            else:
                self._output_plan.joint_references.joint_velocities = (
                    self._output_plan.joint_references.joint_positions - ref_joints
                ) / dt
            self._prev_ref = self._output_plan.joint_references.joint_positions.copy()

        if self._output_plan is None:
            return LocalMotionPlan()

        if self._forward_gp:
            self._output_plan.relative_pose = global_plan.relative_pose
            self._output_plan.end_effector_references = global_plan.end_effector_references
            self._output_plan.twist = global_plan.twist

        return self._output_plan
