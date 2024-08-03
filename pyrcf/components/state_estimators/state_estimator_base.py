"""Base class defining the interface for all state estimators to be used in the control loop."""

from abc import abstractmethod

from ..pyrcf_component import PyRCFComponent
from ...core.types import RobotState


class StateEstimatorBase(PyRCFComponent):  # pylint: disable=too-few-public-methods
    """Base class defining the interface for all state estimators to be used in the control loop."""

    @abstractmethod
    def update_robot_state_with_state_estimates(
        self,
        robot_state: RobotState,
        t: float = None,
        dt: float = None,
    ) -> RobotState:
        """The main state estimator update method to be called in the loop.

        This method will be called after the proprioceptive data in RobotState is retreived by
        calling the `RobotInterface->read()` method.

        This method should be implemented by any state estimator implementation.

        Args:
            robot_state (RobotState): The current state information from the robot.
            t (float, optional): the current time signature of the control loop. Defaults to None
                (controllers may or may not need this).
            dt (float, optional): the time since the last control loop. Defaults to None
                (controllers may or may not need this).

        Raises:
            NotImplementedError: Raised if this method is not implemented by the child class.

        Returns:
            RobotState: Return the same robot state object with the fields for
                robot_state.state_estimates filled appropriately.
        """
        raise NotImplementedError("This method has to be implemented in the child class")


class DummyStateEstimator(StateEstimatorBase):  # pylint: disable=too-few-public-methods
    """Dummy state estimator class for testing pipeline."""

    def __init__(self, squawk: bool = True):
        """Dummy state estimator class for testing pipeline.

        Args:
            squawk (bool, optional): Verbose. Defaults to True.
        """
        self._squawk = squawk

    def update_robot_state_with_state_estimates(
        self,
        robot_state: RobotState,
        t: float = None,
        dt: float = None,
    ) -> RobotState:
        """This is a dummy state estimator that does nothing at all and returns the original state
        object back as is."""
        if self._squawk:
            print(f"{__class__.__name__}: Estimating (fake) robot states.")
        return robot_state
