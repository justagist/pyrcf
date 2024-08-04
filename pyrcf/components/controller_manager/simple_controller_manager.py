from typing import List

from .controller_manager_base import ControllerManagerBase
from ...core.types import GlobalMotionPlan, RobotCmd, RobotState
from ..agents.agent_base import AgentBase
from .command_accumulators.cmd_accumulation_policies import (
    CommandAccumulatorBase,
    SimpleCmdOverride,
)
from ...core.logging import throttled_logging


class SimpleControllerManager(ControllerManagerBase):
    """A simple controller manager that naively performs sequential updates of all agents it
    manages."""

    def __init__(
        self,
        agents: List[AgentBase] = None,
        command_accumulation_policy: CommandAccumulatorBase = SimpleCmdOverride(),
    ):
        """Constructor.

        A simple controller manager that naively performs sequential updates of all agents it
        manages.

        Args:
            agents (List[AgentBase]): List of Agents to be managed by this manager.
            command_accumulation_policy (CommandAccumulatorBase, optional): The command accumulation
                policy to be used on the control outputs. Defaults to SimpleCmdOverride().
        """
        if agents is None:
            agents = []
        super().__init__(agents=agents, command_accumulation_policy=command_accumulation_policy)

    def update(
        self,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        t: float,
        dt: float,
    ) -> RobotCmd:
        if len(self.agents) == 0:
            return RobotCmd()
        if len(self.agents) > 1 and isinstance(self._accumulator, SimpleCmdOverride):
            throttled_logging.warning(
                f"{self.__class__.__name__}: There are multiple agents in "
                "the control loop, but `SimpleCmdOverride` is used as the "
                "command accumulation policy. Only command from last agent "
                "will be used. Use `CmdMuxer` policy instead to comine "
                "commands from all agents.",
                1.0,
            )

        return self._accumulator.accumulate(
            commands=[
                agent.get_action(robot_state=robot_state, global_plan=global_plan, t=t, dt=dt)
                for agent in self.agents
            ]
        )


# NOTE: Ideal controller manager should handle controller switching, starting, stopping etc.
