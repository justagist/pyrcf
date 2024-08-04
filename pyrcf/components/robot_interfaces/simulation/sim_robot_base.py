from abc import abstractmethod
import numpy as np

from ..robot_interface_with_pinocchio import RobotInterfaceWithPinocchio
from ....utils.time_utils import ClockBase
from ....core.types import QuatType


class SimulatedRobotInterface(RobotInterfaceWithPinocchio):
    """Base class for simulated robots.

    All simulated robot classes should also expose a the following methods
    in addition to the standard read() and write() methods:

    1. set_base_pose: This is so that we can use proxy base controlling
        (for mobile manipulators in sim that need stable base motion
        (which may not be ready for testing)).
    2. get_sim_clock: this is to expose the time in the simulated world.
        This can then be used so that all components in the control
        loop can use the same clock from the sim world, and makes the
        velocity measurements in the sim interface more sensible wrt
        the clock being used in the control loop. (see example 05 to
        see how this clock is used in the control loop.)
    """

    @abstractmethod
    def set_base_pose(self, position: np.ndarray, orientation: QuatType):
        """
        Set the pose of the base of the simulated robot in the world frame.

        Args:
            position (np.ndarray): Desired position for the base frame in the world.
            orientation (QuatType): Desired orientation quaternion [x,y,z,w] for base frame
                in the world.

        Raises:
            NotImplementedError: Raised if this method is not implemented by the child class.
        """
        raise NotImplementedError("This method has to be implemented in the child class")

    @abstractmethod
    def get_sim_clock(self) -> ClockBase:
        """
        Get the Clock object that gives the time in this simulated world.

        Raises:
            NotImplementedError: Raised if this method is not implemented by the child class.

        Returns:
            ClockBase: The Clock object that can be used to get the current time using its
                get_time() method.
        """
        raise NotImplementedError("This method has to be implemented in the child class")
