from abc import abstractmethod

from ..pyrcf_component import PyRCFComponent
from ...core.types import RobotState, RobotCmd


class RobotInterface(PyRCFComponent):
    """Base class defining the interface for all robots that can be used in the control loop."""

    @abstractmethod
    def read(self) -> RobotState:
        """Get the latest state from the robot.

        Raises:
            NotImplementedError: Raised if this method is not implemented by the child class.

        Returns:
            RobotState: the latest state of the robot.
        """
        raise NotImplementedError("This method has to be implemented in the child class")

    @abstractmethod
    def write(self, cmd: RobotCmd) -> bool:
        """Send the specified commands to the robot.

        Args:
            cmd (RobotCmd): the command object as computed by the controller in the loop.

        Raises:
            NotImplementedError: Raised if this method is not implemented by the child class.

        Returns:
            bool: True if sending command was succesful.
        """
        raise NotImplementedError("This method has to be implemented in the child class")

    def activate(self) -> bool:
        """This method will be called in a loop until it returns true, before the control loop
        starts. Override in child class as required."""
        return True

    def deactivate(self) -> bool:
        """This method will be called in a loop until it returns true, after the control loop is
        stopped, and before shutdown of this class is called. Override in child class as required.
        """
        return True


class DummyRobot(RobotInterface):
    def __init__(self, squawk: bool = True):
        """Dummy robot class for testing pipeline.

        Args:
            squawk (bool, optional): Verbose. Defaults to True.
        """
        self._squawk = squawk

    def read(self) -> RobotState:
        """This is a dummy method for sanity checking the control loop."""
        if self._squawk:
            print(f"{__class__.__name__}: Reading state from dummy robot")
        return RobotState()

    def write(self, cmd: RobotCmd) -> bool:
        """This is a dummy method for sanity checking the control loop."""
        if self._squawk:
            print(f"{__class__.__name__}: Writing cmd to dummy robot")
        return True
