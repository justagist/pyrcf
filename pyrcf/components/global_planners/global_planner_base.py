from abc import abstractmethod

from ..pyrcf_component import PyRCFComponent
from ...core.types import GlobalMotionPlan, RobotState


class GlobalMotionPlannerBase(PyRCFComponent):
    """Simple protocol and base class for a global motion planner interface.

    This protocol has to be respected by all classes that generate a GlobalMotionPlan object
    to be used in the control loop by local planners. (e.g. user interfaces such as
    keyboard_interface).

    INFO: A general 'global motion planner' is a planner that performs high-level planning that is
        dependent on the task the robot has to perform. For e.g. for a A->B locomotion, the global
        planner might generate a sequence of collision-aware waypoints for the local planner to
        respect.
    """

    @abstractmethod
    def generate_global_plan(
        self, robot_state: RobotState, t: float, dt: float
    ) -> GlobalMotionPlan:
        """Generate a global motion plan message.

        Args:
            robot_state (RobotState): Current robot state.
            t (float, optional): the current time signature of the control loop.
            dt (float, optional): the time since the last control loop.

        Returns:
            GlobalMotionPlan: Output global plan to be sent to local planner in the control loop.
        """
