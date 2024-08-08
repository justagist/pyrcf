"""Utility functions to help with coordinate frame transforms."""

from typing import Tuple, TypeAlias
import numpy as np
from pybullet import multiplyTransforms, invertTransform

from .math_utils import quat2rpy, rpy2quat, vec2skew

QuatType: TypeAlias = np.ndarray
"""Numpy array representating quaternion in format [x,y,z,w]"""

Vector3D: TypeAlias = np.ndarray
"""Numpy array representating 3D cartesian vector in format [x,y,z]"""


def transform_pose_to_frame(
    pos_in_frame_1: Vector3D,
    quat_in_frame_1: QuatType,
    frame_2_pos_in_frame_1: Vector3D,
    frame_2_quat_in_frame_1: QuatType,
) -> Tuple[Vector3D, QuatType]:
    """Transform a pose from a global frame to a local frame.

    Both the input pose and the local frame's pose should be described in a
    common global frame.

    Args:
        pos_in_frame_1 (Vector3D): The position of the input pose object to be transformed in
            a common global frame.
        quat_in_frame_1 (QuatType): The orientation of the input pose object to be transformed in
            a common global frame (quaternion: [x,y,z,w]).
        frame_2_pos_in_frame_1 (Vector3D): Position of the origin of the target coordinate frame
            with respect to the common global frame.
        frame_2_quat_in_frame_1 (QuatType): The orientation of the target frame in the global
            frame.

    Returns:
        Tuple[Vector3D, QuatType]: Position and orientation of the input pose in the new target
            frame.
    """
    p, q = multiplyTransforms(
        *invertTransform(frame_2_pos_in_frame_1, frame_2_quat_in_frame_1),
        pos_in_frame_1,
        quat_in_frame_1,
    )
    return np.array(p), np.array(q)


def twist_transform(twist_a: np.ndarray, frame_b_in_a: np.ndarray) -> np.ndarray:
    """Transform a twist screw (linear velocity & angular velocity) to another coordinate frame.

    Args:
        twist_a (np.ndarray): 6D twist screw (linear x, linear y, linear z, angular x, angular y,
            angular z)
        frame_b_in_a (np.ndarray): Transformation matrix of the pose of the new frame in the old
            frame.

    Returns:
        np.ndarray: Twist screw in the new frame (linear x, linear y, linear z, angular x,
            angular y, angular z).
    """
    rot_mat = frame_b_in_a[0:3, 0:3]
    translation = frame_b_in_a[0:3, 3]
    skew_translation = vec2skew(translation)
    velocity_transform_mat = np.block(
        [
            [rot_mat, np.matmul(skew_translation, rot_mat)],
            [np.zeros((3, 3)), rot_mat],
        ]
    )
    output_twist = np.matmul(velocity_transform_mat, twist_a)
    return output_twist


def get_relative_pose_between_vectors(
    pos1: Vector3D, quat1: QuatType, pos2: Vector3D, quat2: QuatType
) -> Tuple[Vector3D, QuatType]:
    """Given two frame poses in the same coordinate frame, return the pose of
    the second frame with respect to the first.

    Args:
        pos1 (Vector3D): Position of the first frame in a global frame.
        quat1 (QuatType): Orientation quaternion of the first frame.
        pos2 (Vector3D): Position of the second frame in the same global frame.
        quat2 (QuatType): Orientation of the second frame.

    Returns:
        Tuple[Vector3D, QuatType]: Position and orientation of the second frame
            with respect to the first.
    """
    return transform_pose_to_frame(
        pos_in_frame_1=pos2,
        quat_in_frame_1=quat2,
        frame_2_pos_in_frame_1=pos1,
        frame_2_quat_in_frame_1=quat1,
    )


class PoseTrasfrom:
    """Transform a pose (point and orientation) between PyRCF coordinate frames."""

    @staticmethod
    def TELEOP_FRAME_POSE_IN_WORLD(
        base_pos_in_world: Vector3D, base_ori_in_world: QuatType
    ) -> Tuple[Vector3D, QuatType]:
        """This is the definition of the teleop frame."""
        return np.array([base_pos_in_world[0], base_pos_in_world[1], 0.0]), rpy2quat(
            rpy=[0, 0, quat2rpy(base_ori_in_world)[2]]
        )

    @staticmethod
    def BASE_FRAME_POSE_IN_TELEOP(
        base_pos_in_world: Vector3D, base_ori_in_world: QuatType
    ) -> Tuple[Vector3D, QuatType]:
        """This is the definition of the teleop frame."""
        rpy = quat2rpy(base_ori_in_world)
        return np.array([0.0, 0.0, base_pos_in_world[2]]), rpy2quat(rpy=[rpy[0], rpy[1], 0.0])

    @staticmethod
    def teleop2base(
        p_in_teleop: Vector3D,
        q_in_teleop: QuatType,
        base_pos_in_world: Vector3D,
        base_ori_in_world: QuatType,
    ) -> Tuple[Vector3D, QuatType]:
        """Transform a pose in teleop frame to base frame."""
        return transform_pose_to_frame(
            p_in_teleop,
            q_in_teleop,
            *PoseTrasfrom.BASE_FRAME_POSE_IN_TELEOP(base_pos_in_world, base_ori_in_world),
        )

    @staticmethod
    def base2teleop(
        p_in_base: Vector3D,
        q_in_base: QuatType,
        base_pos_in_world: Vector3D,
        base_ori_in_world: QuatType,
    ) -> Tuple[Vector3D, QuatType]:
        """Transform a pose in base frame to teleop frame."""
        return transform_pose_to_frame(
            p_in_base,
            q_in_base,
            *invertTransform(
                *(
                    PoseTrasfrom.BASE_FRAME_POSE_IN_TELEOP(
                        base_pos_in_world,
                        base_ori_in_world,
                    )
                )
            ),
        )

    @staticmethod
    def teleop2world(
        p_in_teleop: Vector3D,
        q_in_teleop: QuatType,
        base_pos_in_world: Vector3D,
        base_ori_in_world: QuatType,
    ) -> Tuple[Vector3D, QuatType]:
        """Transform a pose in teleop frame to world frame."""
        return transform_pose_to_frame(
            p_in_teleop,
            q_in_teleop,
            *invertTransform(
                *PoseTrasfrom.TELEOP_FRAME_POSE_IN_WORLD(base_pos_in_world, base_ori_in_world)
            ),
        )

    @staticmethod
    def world2teleop(
        p_in_world: Vector3D,
        q_in_world: QuatType,
        base_pos_in_world: Vector3D,
        base_ori_in_world: QuatType,
    ) -> Tuple[Vector3D, QuatType]:
        """Transform a pose in world frame to teleop frame."""
        return transform_pose_to_frame(
            p_in_world,
            q_in_world,
            *PoseTrasfrom.TELEOP_FRAME_POSE_IN_WORLD(base_pos_in_world, base_ori_in_world),
        )
