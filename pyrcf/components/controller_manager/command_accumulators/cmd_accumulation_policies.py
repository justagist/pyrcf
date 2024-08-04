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
    def accumulate(self, commands=List[RobotCmd]) -> RobotCmd:
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

    def accumulate(self, commands=List[RobotCmd]) -> RobotCmd:
        return commands[-1]


class CmdMuxer(CommandAccumulatorBase):
    """This policy combines the commands from all controllers/agents."""

    def __init__(self, allow_conflicting_interfaces: bool = False):
        self._cmd: RobotCmd = None
        self._allow_conflicts = allow_conflicting_interfaces

    def accumulate(self, commands=List[RobotCmd]) -> RobotCmd:
        # return super().accumulate(commands)
        if self._cmd is None:
            self._cmd = copy.deepcopy(commands[0])
            if len(commands) > 1:
                for cmd in commands[1:]:
                    self._cmd.extend(cmd, run_checks=True, override_existing=self._allow_conflicts)
