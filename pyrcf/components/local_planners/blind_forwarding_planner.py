from .local_planner_base import LocalPlannerBase
from ...core.types import GlobalMotionPlan, RobotState, LocalMotionPlan, ControlMode


class BlindForwardingPlanner(LocalPlannerBase):
    """Simple forwarding planner which directly forwards the global plan message as local plan."""

    def generate_local_plan(
        self,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        t: float,
        dt: float,
    ) -> LocalMotionPlan:
        return LocalMotionPlan(
            relative_pose=global_plan.relative_pose,
            twist=global_plan.twist,
            joint_references=global_plan.joint_references,
            end_effector_references=global_plan.end_effector_references,
            control_mode=ControlMode.CONTROL,
        )
