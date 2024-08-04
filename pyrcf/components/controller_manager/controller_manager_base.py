from abc import ABC, abstractmethod
from typing import List

from ...core.types import GlobalMotionPlan, RobotCmd, RobotState
from ..agents.agent_base import AgentBase
from .command_accumulators.cmd_accumulation_policies import CommandAccumulatorBase


class ControllerManagerBase(ABC):
    """Base class for writing controller (agent) managers."""

    def __init__(
        self,
        agents: List[AgentBase],
        command_accumulation_policy: CommandAccumulatorBase,
    ):
        """Constructor.

        Args:
            agents (List[AgentBase]): List of Agents to be managed by this manager.
            command_accumulation_policy (CommandAccumulatorBase): The command accumulation
                policy to be used on the control outputs.
        """
        self.agents = agents
        self._accumulator = command_accumulation_policy

    def add_agent(self, agent: AgentBase):
        """Add an agent to be managed by this controller manager.

        Args:
            agent (AgentBase): The agent to be added to the end of the list.

        Returns:
            ControllerManagerBase: The self object with the new agent added.
                This is for using this method in a builder strategy.
        """
        assert isinstance(agent, AgentBase)
        self.agents.append(agent)
        return self

    def set_cmd_accumulation_policy(self, policy: CommandAccumulatorBase):
        """Set/change the command accumulation policy for this mananger.

        Args:
            policy (CommandAccumulatorBase): The new policy to be used.

        Returns:
            ControllerManagerBase: The self object with the policy updated.
                This is for using this method in a builder strategy.
        """
        assert isinstance(policy, CommandAccumulatorBase)
        self._accumulator = policy
        return self

    @abstractmethod
    def update(
        self,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        t: float,
        dt: float,
    ) -> RobotCmd:
        """Run one update step of all the 'agents' managed by this manager.

        Args:
            robot_state (RobotState): Current robot state. This is equivalent to 'observation' in
                standard learning-based agents.
            global_plan (GlobalMotionPlan): The reference global plan to follow.
            t (float, optional): the current time signature of the control loop. Defaults to None
                (controllers may or may not need this).
            dt (float, optional): the time since the last control loop. Defaults to None
                (controllers may or may not need this).

        Returns:
            RobotCmd: The output control command (action) to be sent to the robot based on the
                accumulation policy used by this manager.

        Raises:
            NotImplementedError: Raised if this method is not implemented by the child class.
        """
        raise NotImplementedError("This method has to be implemented in the child class")

    def shutdown(self):
        """Cleanly shutdown all agents managed by this controller manager."""
        for agent in self.agents:
            agent.shutdown()


# NOTE: Ideal controller manager should handle controller switching, starting, stopping etc.
