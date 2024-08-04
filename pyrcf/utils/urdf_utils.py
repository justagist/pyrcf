from typing import Mapping
from contextlib import contextmanager
import os
import tempfile
import yourdfpy
import numpy as np
from scipy.spatial.transform import Rotation


def _convert_to_fixed(urdf: yourdfpy.URDF, joint_names_to_values: Mapping[str, float]):
    """Hack to modify urdf to fix specific joints to specified values.

    Args:
        urdf (yourdfpy.URDF): The yourdfpy urdf object to modify.
        joint_names_to_values (Mapping[str, float]): Dictionary containing name of joints that
            should be locked as keys, and the joint values to lock them at as values.
    """
    robot = urdf.robot
    for joint in robot.joints:
        if joint.name in joint_names_to_values.keys():
            # joints are modified by changing the origin of the
            # joint location wrt to the parent link
            new_local_transformation = np.eye(4)
            if joint.type == "prismatic":
                new_local_transformation[np.where(joint.axis == 1)[0], 3] = joint_names_to_values[
                    joint.name
                ]
            elif joint.type in ["revolute", "continuous"]:
                local_rpy = np.zeros(3)
                local_rpy[np.where(joint.axis == 1)[0]] = joint_names_to_values[joint.name]
                new_local_transformation[:3, :3] = Rotation.from_euler("xyz", local_rpy).as_matrix()
            joint.origin = joint.origin @ new_local_transformation
            joint.type = "fixed"


@contextmanager
def temp_path_urdf_with_joints_fixed(
    urdf_file_path: str, joint_names_to_values: Mapping[str, float]
):
    """Get the context of a temporary URDF file path after modifying the specified joints to be
    "fixed".

    This method only provides a context. This is useful for temporarily writing the URDF to a file
    for consumption by other programs.

    Args:
        urdf_file_path (str): Path to original urdf file.
        joint_names_to_values (Mapping[str, float]): Dictionary containing name of joints that
            should be locked as keys, and the joint values to lock them at as values.

    Yields:
        str: Path to new urdf file.

    Examples
    --------
    Use as a context manager:
        >>> with temp_path_urdf_with_joints_fixed(...) as path:
                load_file_from_path(path)  # do something with this urdf
    """
    urdf = yourdfpy.URDF.load(urdf_file_path)
    _convert_to_fixed(urdf=urdf, joint_names_to_values=joint_names_to_values)

    _, new_urdf_path = tempfile.mkstemp(suffix=".urdf")
    try:
        urdf.write_xml_file(new_urdf_path)
        yield new_urdf_path
    finally:
        os.remove(new_urdf_path)


def create_new_urdf_with_joints_fixed(
    urdf_file_path: str, joint_names_to_values: Mapping[str, float]
) -> str:
    """Creates a new urdf file with the specified joints set to "fixed".

    Args:
        urdf_file_path (str): Path to original urdf file.
        joint_names_to_values (Mapping[str, float]): Dictionary containing name of joints that
            should be locked as keys, and the joint values to lock them at as values.

    Returns:
        str: Path to new urdf file.
    """

    urdf = yourdfpy.URDF.load(urdf_file_path)
    _convert_to_fixed(urdf=urdf, joint_names_to_values=joint_names_to_values)

    new_urdf_path = urdf_file_path[:-5] + "_modified_fixed_joints.urdf"
    urdf.write_xml_file(new_urdf_path)

    return new_urdf_path
