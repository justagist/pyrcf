from typing import TypeAlias, Tuple
import numpy as np
from scipy.spatial.transform import Rotation

QuatType: TypeAlias = np.ndarray
"""Numpy array representating quaternion in format [x,y,z,w]"""
Vector3D: TypeAlias = np.ndarray
"""Numpy array representating 3D cartesian vector in format [x,y,z]"""


def transformation_matrix(
    position: Vector3D = np.array([0, 0, 0]),
    orientation: QuatType = np.array([0, 0, 0, 1]),
) -> np.ndarray:
    """Create a 4x4 transformation matrix

    Args:
        position (Vector3D, optional): position vector. Defaults to np.array([0, 0, 0]).
        orientation (QuatType, optional): orientation quaternion. Defaults to
            np.array([0, 0, 0, 1]).

    Returns:
        np.ndarray: 4x4 transformation matrix representing this pose.
    """
    mat = np.eye(4)
    mat[:3, :3] = quat2rot(quaternion=orientation)
    mat[:3, 3] = position
    return mat


def pos_quat_from_trans_mat(trans_mat: np.ndarray) -> Tuple[Vector3D, QuatType]:
    """Get position and quaternion from a transformation matrix.

    Args:
        trans_mat (np.ndarray): 4x4 transformation matrix.

    Returns:
        Tuple[Vector3D, QuatType]: Equivalent position and quaternion (x,y,z,w)
    """
    return trans_mat[:3, 3], rot2quat(trans_mat[:3, :3])


def invert_transformation_matrix(trans_mat: np.ndarray) -> np.ndarray:
    """Invert the provided homogeneous transformation matrix.

    Args:
        trans_mat (np.ndarray): Input transformation matrix.

    Returns:
        np.ndarray: Inverse of provided transformation matrix.
    """
    rotmat_t = trans_mat[:3, :3].T
    assert is_rotation_matrix(rotmat_t), "Provided matrix is not a valid transformation matrix."
    mat = np.eye(4)
    mat[:3, :3] = rotmat_t
    mat[:3, 3] = -rotmat_t @ trans_mat[:3, 3]
    return mat


def invert_quaternion(quat: QuatType) -> QuatType:
    """Get the quaternion that would give the inverse rotation of the given quaternion.

    Args:
        quat (QuatType): Input quaternion (x,y,z,w)

    Returns:
        QuatType: Inverse of the given quaternion (x,y,z,w)
    """
    out_q = quat.copy()
    out_q[:3] *= -1
    return out_q / np.linalg.norm(out_q)


def quat_error(quat1: QuatType, quat2: QuatType) -> float:
    """Compute shortest angle (in radians) between two quaternions.

    Args:
        quat1 (QuatType): Input quaternion 1. Order: (x,y,z,w)
        quat2 (QuatType): Input quaternion 2. Order: (x,y,z,w)

    Returns:
        float: absoluted error in radians (between 0 and pi)
    """
    return (Rotation.from_quat(quat1) * Rotation.from_quat(quat2).inv()).magnitude()


def quat_diff(quat1: QuatType, quat2: QuatType) -> QuatType:
    """Get the difference quaternion between two quaternions.

    Args:
        quat1 (QuatType): Input quaternion 1 (final). Order: (x,y,z,w)
        quat2 (QuatType): Input quaternion 2 (initial). Order: (x,y,z,w)

    Returns:
        QuatType: Output difference quaternion. Order: (x,y,z,w)
    """
    return (Rotation.from_quat(quat1) * Rotation.from_quat(quat2).inv()).as_quat()


def rpy_error(rpy1: Vector3D, rpy2: QuatType) -> float:
    """Compute shortest angle (in radians) between two rotations (described by roll-pitch-yaw).

    Args:
        rpy1 (QuatType): Input RPY value 1
        rpy2 (QuatType): Input RPY value 2

    Returns:
        float: absoluted error in radians (between 0 and pi)
    """
    return (Rotation.from_euler("xyz", rpy1) * Rotation.from_euler("xyz", rpy2).inv()).magnitude()


def rotmat_error(rotmat1: np.ndarray, rotmat2: np.ndarray):
    """Compute shortest angle (in radians) between two rotations (described by rotation matrices).

    Args:
        rotmat1 (QuatType): Input rotation matrix 1
        rotmat2 (QuatType): Input rotation matrix 2

    Returns:
        float: absoluted error in radians (between 0 and pi)
    """
    return (Rotation.from_matrix(rotmat1) * Rotation.from_matrix(rotmat2).inv()).magnitude()


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap the provided angle(s) (radians) to be within -pi and pi.

    Args:
        angle (float | np.ndarray): Input angle (or array of angles) in radians.

    Returns:
        float | np.ndarray: Output after wrapping.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def quat_multiply(quat1: QuatType, quat2: QuatType) -> QuatType:
    """Multiply quat1 with quat2. (q1 * q2)

    Args:
        quat1 (QuatType): Input quaternion 1
        quat2 (QuatType): Input quaternion 2

    Returns:
        QuatType: Output quaternion (q1*q2)
    """
    return (Rotation.from_quat(quat1) * Rotation.from_quat(quat2)).as_quat()


def quat2rpy(quaternion: QuatType) -> np.ndarray:
    """Convert a quaternion into euler angles (roll, pitch, yaw).

    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)

    Args:
        quaternion (QuatType): x,y,z,w order

    Returns:
        np.ndarray: Euler engles in rpy order.
    """
    try:
        return Rotation.from_quat(quaternion).as_euler("xyz")
    except ValueError as e:
        raise ValueError(f"{e}: Culprit: {quaternion}") from e


def rpy2quat(rpy: np.ndarray) -> QuatType:
    """Convert an Euler angle to a quaternion.

    Args:
        rpy (np.ndarray): roll, pitch, yaw in radians

    Returns:
        QuatType: The orientation in quaternion [x,y,z,w] format
    """
    return Rotation.from_euler("xyz", rpy).as_quat()


def xyz_local2quat(xyz: np.ndarray) -> QuatType:
    """Convert sequential local x,y,z rotations to a quaternion.

    Args:
        xyz (np.ndarray): orientations along x, y, z (local) in radians

    Returns:
        QuatType: The orientation in quaternion [x,y,z,w] format
    """
    xyz = np.array(xyz).reshape([-1, 3])
    zyx = xyz.copy()
    zyx[:, 0] = xyz[:, 2]
    zyx[:, 2] = xyz[:, 0]
    q = Rotation.from_euler("zyx", zyx).as_quat()
    if q.shape[0] == 1:
        return q.flatten()
    return q


def quat2xyz_local(quat: QuatType) -> np.ndarray:
    """Convert quaternion to sequential local x,y,z rotations.

    Args:
        quat (QuatType): The orientation in quaternion [x,y,z,w] format

    Returns:
        np.ndarray: orientations along x, y, z (local) in radians. Shape: (3,)
    """
    zyx = Rotation.from_quat(quat).as_euler("zyx").reshape([-1, 3])
    xyz = zyx.copy()
    xyz[:, 0] = zyx[:, 2]
    xyz[:, 2] = zyx[:, 0]
    if xyz.shape[0] == 1:
        return xyz.flatten()
    return xyz


def quat2rot(quaternion: QuatType) -> np.ndarray:
    """Covert a quaternion into a full three-dimensional rotation matrix.

    Args:
        quaternion (QuatType): A 4 element array representing the quaternion (w,x,y,z)

    Returns:
        np.ndarray: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    try:
        return Rotation.from_quat(quaternion).as_matrix()
    except ValueError:
        # BUG: occasional nans in quaternion retrieved from pybullet
        # https://github.com/bulletphysics/bullet3/issues/976
        print(f"Quaternion error: {quaternion}")
        raise


def rpy2rot(rpy: np.ndarray) -> np.ndarray:
    """Converts roll, pitch, yaw euler angles to 3x3 rotation matrix.

    Args:
        rpy (np.ndarray): roll, pitch, yaw in radians

    Returns:
        np.ndarray: equivalent 3x3 rotation matrix
    """
    return Rotation.from_euler("xyz", rpy).as_matrix()


def is_rotation_matrix(mat: np.ndarray) -> bool:
    """Checks if a matrix is a valid rotation matrix.

    Args:
        mat (np.ndarray): Input matrix to check

    Returns:
        bool: True if this is a valid rotation matrix.
    """
    return np.linalg.norm(np.eye(3) - np.dot(np.array(mat).T, mat)) < 1e-6


def rot2rpy(rotmat: np.ndarray) -> np.ndarray:
    """Converts 3x3 rotation matrix to euler roll, pitch, yaw angles.

    Args:
        rotmat (np.ndarray): 3x3 rotation matrix.

    Returns:
        np.ndarray: roll, pitch, yaw in radians.
    """
    assert is_rotation_matrix(rotmat), "Provided R matrix is not a valid rotation matrix."
    return Rotation.from_matrix(rotmat).as_euler("xyz")


def rot2quat(rotmat: np.ndarray) -> QuatType:
    """Converts 3x3 rotation matrix to quaternion.

    Args:
        rotmat (np.ndarray): 3x3 rotation matrix.

    Returns:
        QuatType: Quaternion (x,y,z,w)
    """
    assert is_rotation_matrix(rotmat), "Provided R matrix is not a valid rotation matrix."
    return Rotation.from_matrix(rotmat).as_quat()


def vec2skew(vector: Vector3D) -> np.ndarray:
    """Convert provided vector to its equivalent skew-symmetric matrix.

    Args:
        vector (Vector3D): 3d vector.

    Returns:
        np.ndarray: 3x3 skew symmetric matrix.
    """
    skew_mat = np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )

    return skew_mat


def vec2quatskew(vector: Vector3D) -> np.ndarray:
    """Convert provided vector to its "Omega" operator as described in
    https://ahrs.readthedocs.io/en/latest/filters/angular.html.

    Args:
        vector (Vector3D): 3d vector.

    Returns:
        np.ndarray: 4x4 Omega operator matrix.
    """

    omega_matrix = np.array(
        [
            [0, -vector[0], -vector[1], -vector[2]],
            [vector[0], 0, vector[2], -vector[1]],
            [vector[1], -vector[2], 0, vector[0]],
            [vector[2], vector[1], -vector[0], 0],
        ]
    )

    return omega_matrix
