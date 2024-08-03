import pytest
import numpy as np
from pyrcf.core.types import Pose3D


class TestPose3D:

    @pytest.fixture(scope="class")
    def valid_position_orientation(self):
        return np.array([1, 2, 3]), np.array([0, 0, 0, 1])

    @pytest.fixture(scope="class")
    def invalid_position_orientation(self):
        return np.array([1, 2, 3, 4]), np.array([1, 0, 0, 1])

    def test_valid_pose(self, valid_position_orientation):
        _ = Pose3D(
            position=valid_position_orientation[0],
            orientation=valid_position_orientation[1],
            validate_quaternion=True,
        )

    def test_invalid_position(self, valid_position_orientation, invalid_position_orientation):
        with pytest.raises(AssertionError, match="Position dimension should be 3."):
            _ = Pose3D(
                position=invalid_position_orientation[0],
                orientation=valid_position_orientation[1],
                validate_quaternion=True,
            )
