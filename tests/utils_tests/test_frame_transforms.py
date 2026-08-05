"""Tests for `pyrcf.utils.frame_transforms`.

This module used pybullet's `multiplyTransforms`/`invertTransform` for what is pure rigid-transform
maths; it now uses scipy-backed helpers from `math_utils`. These tests pin the behaviour down with
round trips and known-geometry cases so the swap (and any future one) is checked.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyrcf.utils.frame_transforms import (
    PoseTrasfrom,
    get_relative_pose_between_vectors,
    transform_pose_to_frame,
    twist_transform,
)
from pyrcf.utils.math_utils import quat_error, transformation_matrix

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def quat_close(q1, q2, atol=1e-9):
    return np.allclose(q1, q2, atol=atol) or np.allclose(q1, -np.asarray(q2), atol=atol)


class TestTransformPoseToFrame:

    def test_identity_target_frame_is_a_no_op(self):
        pos, quat = np.array([1.0, 2.0, 3.0]), Rotation.from_euler("z", 0.4).as_quat()
        p, q = transform_pose_to_frame(pos, quat, np.zeros(3), IDENTITY_QUAT)
        assert np.allclose(p, pos)
        assert quat_close(q, quat)

    def test_pure_translation(self):
        p, q = transform_pose_to_frame(
            np.array([1.0, 0.0, 0.0]), IDENTITY_QUAT, np.array([1.0, 1.0, 0.0]), IDENTITY_QUAT
        )
        assert np.allclose(p, [0.0, -1.0, 0.0])
        assert quat_close(q, IDENTITY_QUAT)

    def test_pure_rotation_of_the_target_frame(self):
        """A point on +x, seen from a frame rotated +90 deg about z, lies on -y."""
        p, _ = transform_pose_to_frame(
            np.array([1.0, 0.0, 0.0]),
            IDENTITY_QUAT,
            np.zeros(3),
            Rotation.from_euler("z", np.pi / 2).as_quat(),
        )
        assert np.allclose(p, [0.0, -1.0, 0.0], atol=1e-12)

    def test_agrees_with_a_homogeneous_matrix_reference(self):
        rng = np.random.default_rng(0)
        quats = Rotation.random(40, random_state=1).as_quat()
        frame_quats = Rotation.random(40, random_state=2).as_quat()
        for i in range(40):
            pos, quat = rng.normal(size=3), quats[i]
            f_pos, f_quat = rng.normal(size=3), frame_quats[i]
            expected = np.linalg.inv(transformation_matrix(f_pos, f_quat)) @ transformation_matrix(
                pos, quat
            )
            p, q = transform_pose_to_frame(pos, quat, f_pos, f_quat)
            assert np.allclose(p, expected[:3, 3], atol=1e-12)
            assert quat_error(q, Rotation.from_matrix(expected[:3, :3]).as_quat()) < 1e-9

    def test_round_trip_back_to_the_original_frame(self):
        rng = np.random.default_rng(3)
        for quat, f_quat in zip(
            Rotation.random(20, random_state=4).as_quat(),
            Rotation.random(20, random_state=5).as_quat(),
            strict=True,
        ):
            pos, f_pos = rng.normal(size=3), rng.normal(size=3)
            p2, q2 = transform_pose_to_frame(pos, quat, f_pos, f_quat)
            # transforming back requires the inverse frame pose
            inv = np.linalg.inv(transformation_matrix(f_pos, f_quat))
            p3, q3 = transform_pose_to_frame(
                p2, q2, inv[:3, 3], Rotation.from_matrix(inv[:3, :3]).as_quat()
            )
            assert np.allclose(p3, pos, atol=1e-10)
            assert quat_error(q3, quat) < 1e-9


class TestTwistTransform:

    def test_identity_frame_is_a_no_op(self):
        twist = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        assert np.allclose(twist_transform(twist, np.eye(4)), twist)

    def test_pure_rotation_rotates_both_halves(self):
        twist = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        frame = transformation_matrix(np.zeros(3), Rotation.from_euler("z", np.pi / 2).as_quat())
        out = twist_transform(twist, frame)
        assert out.shape == (6,)
        # magnitudes are preserved by a pure rotation
        assert np.linalg.norm(out[:3]) == pytest.approx(1.0, abs=1e-9)
        assert np.linalg.norm(out[3:]) == pytest.approx(1.0, abs=1e-9)

    def test_angular_magnitude_is_preserved_under_pure_translation(self):
        twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        frame = transformation_matrix(np.array([1.0, 0.0, 0.0]), IDENTITY_QUAT)
        out = twist_transform(twist, frame)
        assert np.linalg.norm(out[3:]) == pytest.approx(1.0, abs=1e-9)


class TestRelativePoseBetweenFrames:
    """`get_relative_pose_between_vectors` actually takes two full *poses* (pos+quat each) and
    returns the second expressed relative to the first, despite the name."""

    def test_identical_poses_give_zero_relative_pose(self):
        pos, quat = np.array([1.0, -2.0, 0.5]), Rotation.from_euler("y", 0.3).as_quat()
        p, q = get_relative_pose_between_vectors(pos, quat, pos, quat)
        assert np.allclose(p, np.zeros(3), atol=1e-12)
        assert quat_error(q, IDENTITY_QUAT) == pytest.approx(0.0, abs=1e-9)

    def test_pure_offset_between_aligned_frames(self):
        p, q = get_relative_pose_between_vectors(
            np.zeros(3), IDENTITY_QUAT, np.array([1.0, 2.0, 3.0]), IDENTITY_QUAT
        )
        assert np.allclose(p, [1.0, 2.0, 3.0])
        assert quat_error(q, IDENTITY_QUAT) == pytest.approx(0.0, abs=1e-9)

    def test_matches_transform_pose_to_frame(self):
        rng = np.random.default_rng(6)
        q1s = Rotation.random(30, random_state=7).as_quat()
        q2s = Rotation.random(30, random_state=8).as_quat()
        for q1, q2 in zip(q1s, q2s, strict=True):
            p1, p2 = rng.normal(size=3), rng.normal(size=3)
            got = get_relative_pose_between_vectors(p1, q1, p2, q2)
            expected = transform_pose_to_frame(p2, q2, p1, q1)
            assert np.allclose(got[0], expected[0], atol=1e-12)
            assert quat_close(got[1], expected[1])

    def test_relative_orientation_recovers_a_known_angle(self):
        q1 = IDENTITY_QUAT
        q2 = Rotation.from_euler("z", 0.6).as_quat()
        _, q = get_relative_pose_between_vectors(np.zeros(3), q1, np.zeros(3), q2)
        assert quat_error(q, q2) == pytest.approx(0.0, abs=1e-9)

    def test_is_antisymmetric_in_position(self):
        rng = np.random.default_rng(9)
        p1, p2 = rng.normal(size=3), rng.normal(size=3)
        fwd, _ = get_relative_pose_between_vectors(p1, IDENTITY_QUAT, p2, IDENTITY_QUAT)
        bwd, _ = get_relative_pose_between_vectors(p2, IDENTITY_QUAT, p1, IDENTITY_QUAT)
        assert np.allclose(fwd, -bwd, atol=1e-12)


class TestPoseTransformStatics:
    """Frame-to-frame helpers used by the teleop UI path."""

    BASE_POS = np.array([1.0, 2.0, 0.4])
    BASE_QUAT = Rotation.from_euler("z", 0.7).as_quat()

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_base_teleop_round_trip(self, seed):
        rng = np.random.default_rng(seed)
        pos = rng.normal(size=3)
        quat = Rotation.random(random_state=seed + 10).as_quat()
        pt, qt = PoseTrasfrom.base2teleop(pos, quat, self.BASE_POS, self.BASE_QUAT)
        pb, qb = PoseTrasfrom.teleop2base(pt, qt, self.BASE_POS, self.BASE_QUAT)
        assert np.allclose(pb, pos, atol=1e-10)
        assert quat_error(qb, quat) < 1e-9

    @pytest.mark.parametrize("seed", [3, 4, 5])
    def test_teleop_world_round_trip(self, seed):
        rng = np.random.default_rng(seed)
        pos = rng.normal(size=3)
        quat = Rotation.random(random_state=seed + 20).as_quat()
        pw, qw = PoseTrasfrom.teleop2world(pos, quat, self.BASE_POS, self.BASE_QUAT)
        pt, qt = PoseTrasfrom.world2teleop(pw, qw, self.BASE_POS, self.BASE_QUAT)
        assert np.allclose(pt, pos, atol=1e-10)
        assert quat_error(qt, quat) < 1e-9

    def test_teleop_frame_ignores_base_roll_pitch_and_height(self):
        """The teleop frame is defined with roll, pitch and z fixed, so a pose expressed in it must
        not change when the base pitches or changes height."""
        pos = np.array([0.5, 0.0, 0.2])
        # NOTE: built with the same "xyz" convention `quat2rpy` uses, and the SAME yaw, so that
        # only roll/pitch differ -- the teleop frame does track base yaw by definition.
        upright = Rotation.from_euler("xyz", [0.0, 0.0, 0.3]).as_quat()
        pitched = Rotation.from_euler("xyz", [0.1, 0.25, 0.3]).as_quat()

        a = PoseTrasfrom.teleop2world(pos, IDENTITY_QUAT, np.array([1.0, 2.0, 0.4]), upright)
        b = PoseTrasfrom.teleop2world(pos, IDENTITY_QUAT, np.array([1.0, 2.0, 0.9]), pitched)
        assert np.allclose(
            a[0], b[0], atol=1e-9
        ), "teleop frame should ignore base roll/pitch/height"
