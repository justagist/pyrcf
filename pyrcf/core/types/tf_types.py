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

    position: Vector3D = field(default_factory=lambda: np.zeros(3))
    orientation: QuatType = field(default_factory=lambda: np.array([0, 0, 0, 1]))
    """quaternion [x,y,z,w]"""

    validate_quaternion: bool = field(default=False, init=True, repr=False)
    """If this is enabled, quaternion validity will be checked after construction. Default False."""

    def __eq__(self, other: "Pose3D") -> bool:
        return np.allclose(self.position, other.position) and np.allclose(
            self.orientation, other.orientation
        )

    def __setattr__(self, prop, val):
        match prop:
            case "position":
                assert len(val) == 3, "Position dimension should be 3."
            case "orientation":
                assert len(val) == 4, "Quaternion dimension should be 4."
                if self.validate_quaternion:
                    assert np.isclose(np.linalg.norm(val), 1.0), "Quaternion is not normalised."
            case "validate_quaternion":
                super().__setattr__(prop, val)
                return
            case _:
                raise NameError(
                    f"Tried to assign value to illegal parameter {prop} in object of type "
                    f"{self.__class__.__name__}."
                )
        super().__setattr__(prop, np.array(val))


@dataclass
class Twist:
    """Container for holding velocity in the base frame of the robot.

    This object also allows equality comparison (e.g. check for (`if twist1 == twist2:`)).
    """

    linear: Vector3D = field(default_factory=lambda: np.zeros(3))
    angular: Vector3D = field(default_factory=lambda: np.zeros(3))

    def __eq__(self, other: "Twist") -> bool:
        return np.allclose(self.linear, other.linear) and np.allclose(self.angular, other.angular)

    def __setattr__(self, __name: str, __value: np.ndarray | Tuple[float]) -> None:
        if __name not in ["linear", "angular"]:
            raise NameError(
                f"Tried to assign value to illegal parameter {__name} in object of type "
                f"{self.__class__.__name__}."
            )
        assert len(__value) == 3, f"{__name} velocity should be of dimension 3"
        super().__setattr__(__name, np.array(__value))


@dataclass
class RelativePose:
    """This is used for representing relative pose (typically desired body pose commands) where RPY
    is better.

    This object also allows equality comparison (e.g. check for (`if r_pose1 == r_pose2:`)).
    """

    position: Vector3D = field(default_factory=lambda: np.zeros(3))
    rpy: Vector3D = field(default_factory=lambda: np.zeros(3))
    """[Roll, Pitch, Yaw] in radians"""

    def __eq__(self, other: "RelativePose") -> bool:
        return np.allclose(self.position, other.position) and np.allclose(self.rpy, other.rpy)

    def __setattr__(self, __name: str, __value: np.ndarray | Tuple[float]) -> None:
        if __name not in ["position", "rpy"]:
            raise NameError(
                f"Tried to assign value to illegal parameter {__name} in object of type "
                f"{self.__class__.__name__}."
            )
        assert len(__value) == 3, f"{__name} should be of dimension 3"
        super().__setattr__(__name, np.array(__value))
