"""
Base class for creating user interfaces (UI) to act as fake Global Plan generators.

This interface is for generating a fake global plan to be used by local planners from user
input and 'translating' the input to references that can be used by local planners for generating
control references.

In fully autonomous systems, this will be replaced by a true global planner.

e.g. read joystick input and translate to base velocity commands (as GlobalMotionPlan message)
"""

from abc import abstractmethod

from ..global_planner_base import GlobalMotionPlannerBase
from ....core.types import GlobalMotionPlan, RobotState


class UIBase(GlobalMotionPlannerBase):

    @abstractmethod
    def process_user_input(
        self, robot_state: RobotState, t: float = None, dt: float = None
    ) -> GlobalMotionPlan:
        """Processes user input and returns a GlobalMotionPlan object.

        The method that should be implemented by child classes where a user input is processed
        and appropriate GlobalMotionPlan is returned.

        Returns:
            GlobalMotionPlan: the generated motion plan object.
        """
        raise NotImplementedError("This method has to be implemented in the child class")

    def generate_global_plan(
        self, robot_state: RobotState, t: float = None, dt: float = None
    ) -> GlobalMotionPlan:
        """This method is following the global planner protocol.

        Internally this just calls the `process_user_input()` method.

        Returns:
            GlobalMotionPlan: the generated motion plan object.
        """
        return self.process_user_input(robot_state=robot_state, t=t, dt=dt)


class DummyUI(UIBase):
    def __init__(self, squawk: bool = True):
        """Dummy UI/Global planner class for testing pipeline.

        Args:
            squawk (bool, optional): Verbose. Defaults to True.
        """
        self._squawk = squawk

    def process_user_input(
        self, robot_state: RobotState, t: float = None, dt: float = None
    ) -> GlobalMotionPlan:
        if self._squawk:
            print(f"{__class__.__name__}: Faking user input read step.")
        return GlobalMotionPlan()
