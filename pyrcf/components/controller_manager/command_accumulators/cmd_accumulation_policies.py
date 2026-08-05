"""Accumulation policies are functions that takes in commands from two controllers
and decides how to generate a single command to be sent to the robot.

NOTE: this should ideally be done at joint level, but we are simplifying by doing
accumulation at robot-level instead.
"""

from typing import List
from abc import ABC, abstractmethod
import copy

from ....core.types.robot_io import RobotCmd


class CommandAccumulatorBase(ABC):
    """Accumulation policies are functions that takes in commands from two controllers
    and decides how to generate a single command to be sent to the robot.

    NOTE: this should ideally be done at joint level, but we are simplifying by doing
    accumulation at robot-level instead."""

    @abstractmethod
    def accumulate(self, commands: List[RobotCmd]) -> RobotCmd:
        """Accumulate the specified commands into a single command to be sent to the robot.

        Args:
            commands (List[RobotCmd]): the robot commands from different controllers (in sequence).

        Returns:
            RobotCmd: The final control command (action) to be sent to the robot.

        Raises:
            NotImplementedError: Raised if this method is not implemented by the child class.
        """
        raise NotImplementedError("This method has to be implemented in the child class")


class SimpleCmdOverride(CommandAccumulatorBase):
    """This accumulation policy simply overrides the command from all controllers with the latest
    one."""

    def accumulate(self, commands: List[RobotCmd]) -> RobotCmd:
        return commands[-1]


class CmdMuxer(CommandAccumulatorBase):
    """This policy combines the commands from all controllers/agents."""

    def __init__(self, allow_conflicting_interfaces: bool = False):
        """Constructor.

        Args:
            allow_conflicting_interfaces (bool, optional): If set to True, agents are allowed to
                command the same joint; the command from the later agent in the sequence wins.
                If False (default), an agent commanding a joint that an earlier agent already
                commands raises an AssertionError. Defaults to False.
        """
        self._allow_conflicts = allow_conflicting_interfaces

    def accumulate(self, commands: List[RobotCmd]) -> RobotCmd:
        # NOTE: the accumulated command has to be rebuilt from scratch on every call; caching it
        # across control loop iterations would keep sending the first iteration's command forever.
        accumulated = copy.deepcopy(commands[0])
        for cmd in commands[1:]:
            accumulated.extend(other=cmd, run_checks=True, overwrite_existing=self._allow_conflicts)
        return accumulated
