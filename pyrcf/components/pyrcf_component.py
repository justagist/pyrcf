"""
Abstract base class for all core components in PyRCF control loop.

    Components that inherit this are:
        - RobotInterfaceBase
        - GlobalMotionPlannerBase
        - LocalMotionPlannerBase
        - ControllerBase
        - StateEstimatorBase
"""

from abc import ABC
from ..core.logging import logging


class PyRCFComponent(ABC):
    """Abstract base class for all core components in PyRCF control loop.

    Components that inherit this are:
        - RobotInterfaceBase
        - GlobalMotionPlannerBase
        - LocalMotionPlannerBase
        - ControllerBase
        - StateEstimatorBase
    """

    def shutdown(self):
        """Cleanly shutdown the PyRCF component. Override in child class if required. The
        base class implements an empty function."""
        logging.info(f"{self.__class__.__name__}: Shutting down.")

    def get_class_name(self) -> str:
        """Get the name of the class/type of this object."""
        return self.__class__.__name__

    def get_class_info(self) -> str:
        """Override with custom string if needed."""
        return self.get_class_name()
