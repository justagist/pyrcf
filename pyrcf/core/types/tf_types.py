from typing import TypeAlias, Tuple
from dataclasses import dataclass, field
import numpy as np

QuatType: TypeAlias = np.ndarray  # TODO: maybe use quaternionic package here
"""Numpy array representating quaternion in format [x,y,z,w]"""

Vector3D: TypeAlias = np.ndarray
"""Numpy array representating 3D cartesian vector in format [x,y,z]"""


@dataclass
class Pose3D:
    """Container for holding 3D pose values.

    This object also allows equality comparison (e.g. check for (`if pose1 == pose2:`)).

    NOTE: When creating this object, use `validate_quaternion=True` to assert quaternion validity.
        Defaults to False.
    """

    position: Vector3D = np.zeros(3)
    orientation: QuatType = np.array([0, 0, 0, 1])
    """quaternion [x,y,z,w]"""

    validate_quaternion: bool = field(default=False, init=True, repr=False)
    """If this is enabled, quaternion validity will be checked after construction. Default False."""

    def __post_init__(self):
        self.__do_checks()

    def __do_checks(self):
        assert len(self.position) == 3, "Position dimension should be 3."
        assert len(self.orientation) == 4, "Quaternion dimension should be 4."
        if not self.validate_quaternion:
            return
        assert np.isclose(np.linalg.norm(self.orientation), 1.0), "Quaternion is not normalised."

    def __eq__(self, other: "Pose3D") -> bool:
        return np.allclose(self.position, other.position) and np.allclose(
            self.orientation, other.orientation
        )

    def __setattr__(self, prop, val):
        if prop not in ["position", "orientation", "validate_quaternion"]:
            raise NameError(
                f"Tried to assign value to illegal parameter {prop} in object of type "
                f"{self.__class__.__name__}."
            )
        if prop != "validate_quaternion":
            val = np.array(val)
        super().__setattr__(prop, val)
        self.__do_checks()


@dataclass
class Twist:
    """Container for holding velocity in the base frame of the robot.

    This object also allows equality comparison (e.g. check for (`if twist1 == twist2:`)).
    """

    linear: Vector3D = np.zeros(3)
    angular: Vector3D = np.zeros(3)

    def __post_init__(self):
        self.__do_checks()

    def __do_checks(self):
        assert len(self.linear) == 3, "Linear velocity should be of dimension 3"
        assert len(self.angular) == 3, "Angular velocity should be of dimension 3"

    def __eq__(self, other: "Twist") -> bool:
        return np.allclose(self.linear, other.linear) and np.allclose(self.angular, other.angular)

    def __setattr__(self, __name: str, __value: np.ndarray | Tuple[float]) -> None:
        if __name not in ["linear", "angular"]:
            raise NameError(
                f"Tried to assign value to illegal parameter {__name} in object of type "
                f" {self.__class__.__name__}."
            )
        super().__setattr__(__name, np.array(__value))
        self.__do_checks()


@dataclass
class RelativePose:
    """This is used for representing relative pose (typically desired body pose commands) where RPY
    is better.

    This object also allows equality comparison (e.g. check for (`if r_pose1 == r_pose2:`)).
    """

    position: Vector3D = np.zeros(3)
    rpy: Vector3D = np.zeros(3)
    """[Roll, Pitch, Yaw] in radians"""

    def __post_init__(self):
        self.__do_checks()

    def __do_checks(self):
        assert len(self.position) == 3, "Position should be of dimension 3"
        assert len(self.rpy) == 3, "RPY should be of dimension 3"

    def __eq__(self, other: "RelativePose") -> bool:
        return np.allclose(self.position, other.position) and np.allclose(self.rpy, other.rpy)

    def __setattr__(self, __name: str, __value: np.ndarray | Tuple[float]) -> None:
        if __name not in ["position", "rpy"]:
            raise NameError(
                f"Tried to assign value to illegal parameter {__name} in object of type "
                f"{self.__class__.__name__}."
            )
        super().__setattr__(__name, np.array(__value))
        self.__do_checks()
