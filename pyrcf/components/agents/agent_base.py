from abc import abstractmethod
from typing import Tuple

from ..pyrcf_component import PyRCFComponent
from ...core.types import RobotState, RobotCmd, GlobalMotionPlan, LocalMotionPlan


class AgentBase(PyRCFComponent):
    """An agent is an entity that observes the state of the robot and responds with a control
    action.

    With this in mind, an agent for classical control cases would be an entity that combines the
    functions of a 'local planner' and 'controller' into a single entity called 'Agent', which
    observes the state of the robot interface and responds by sending commands to the robot, so as
    to follow the objective of following the 'global plan' from the global planner.
    """

    @abstractmethod
    def get_action(
        self,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        t: float,
        dt: float,
    ) -> RobotCmd:
        """Compute the control command to be executed given the current state and the global plan.

        Args:
            robot_state (RobotState): Current robot state. This is equivalent to 'observation' in
                standard learning-based agents.
            global_plan (GlobalMotionPlan): The reference global plan to follow.
            t (float, optional): the current time signature of the control loop. Defaults to None
                (controllers may or may not need this).
            dt (float, optional): the time since the last control loop. Defaults to None
                (controllers may or may not need this).

        Returns:
            RobotCmd: The output control command (action) to be sent to the robot.

        Raises:
            NotImplementedError: Raised if this method is not implemented by the child class.
        """
        raise NotImplementedError("This method has to be implemented in the child class")

    @abstractmethod
    def get_last_output(self) -> Tuple[LocalMotionPlan, RobotCmd]:
        """Should return the last computed outputs by this agent.

        Returns:
            Tuple[LocalMotionPlan, RobotCmd]: Output local plan and control command from this agent.
                NOTE: Either of these can be None, if the agent does not compute them.
        """
        raise NotImplementedError("This method has to be implemented in the child class")


class DummyAgent(AgentBase):
    """Dummy Agent for testing pipeline."""

    def __init__(self, squawk: bool = True):
        """Dummy Agent for testing pipeline.

        Args:
            squawk (bool, optional): Verbosity. Defaults to True.
        """
        self._squawk = squawk

    def get_action(
        self,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        t: float,
        dt: float,
    ) -> RobotCmd:
        """This is a dummy method for sanity checking the control loop."""
        if self._squawk:
            print(f"{__class__.__name__}: Performing dummy agent update. time: {t}, dt: {dt}")
        return RobotCmd()

    def get_last_output(self) -> Tuple[LocalMotionPlan | RobotCmd]:
        return None, RobotCmd()
