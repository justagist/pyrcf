"""Tests for the pure rotation/transform maths in `pyrcf.utils.math_utils`.

These are property-based where possible (round trips, agreement with an independent float64
reference) rather than hard-coded expected values, so they stay meaningful if an implementation is
swapped for a different library.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyrcf.utils.math_utils import (
    invert_quaternion,
    invert_transform,
    invert_transformation_matrix,
    is_rotation_matrix,
    multiply_transforms,
    pos_quat_from_trans_mat,
    quat2rot,
    quat2rpy,
    quat_error,
    quat_multiply,
    rot2quat,
    rot2rpy,
    rpy2quat,
    rpy2rot,
    transformation_matrix,
    vec2skew,
    wrap_angle,
)

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def random_quats(n, seed=0):
    """Uniformly distributed random unit quaternions.

    NOTE: built from normalised Gaussians rather than `Rotation.random(random_state=...)`, whose
    signature differs across the scipy versions this project supports (`random_state` was replaced
    by `rng`).
    """
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(n, 4))
    return quats / np.linalg.norm(quats, axis=1, keepdims=True)


def random_positions(n, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 3))


def quat_close(q1, q2, atol=1e-12):
    """Quaternions q and -q are the same rotation."""
    return np.allclose(q1, q2, atol=atol) or np.allclose(q1, -np.asarray(q2), atol=atol)


class TestQuaternionConversions:

    def test_quat_rot_round_trip(self):
        for q in random_quats(50):
            assert quat_close(rot2quat(quat2rot(q)), q, atol=1e-10)

    def test_quat_rpy_round_trip(self):
        for q in random_quats(50, seed=1):
            assert quat_close(rpy2quat(quat2rpy(q)), q, atol=1e-10)

    def test_rpy_rot_round_trip(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            # keep pitch away from +-pi/2 to avoid gimbal lock in the euler representation
            rpy = np.array(
                [
                    rng.uniform(-np.pi, np.pi),
                    rng.uniform(-1.2, 1.2),
                    rng.uniform(-np.pi, np.pi),
                ]
            )
            assert np.allclose(rot2rpy(rpy2rot(rpy)), rpy, atol=1e-10)

    def test_quat2rot_returns_a_valid_rotation_matrix(self):
        for q in random_quats(20, seed=3):
            rot = quat2rot(q)
            assert is_rotation_matrix(rot)
            assert np.isclose(np.linalg.det(rot), 1.0)

    def test_identity_quaternion_maps_to_identity_matrix(self):
        assert np.allclose(quat2rot(IDENTITY_QUAT), np.eye(3))

    def test_is_rotation_matrix_rejects_non_rotations(self):
        assert not is_rotation_matrix(np.eye(3) * 2.0)
        assert not is_rotation_matrix(np.ones((3, 3)))


class TestQuaternionAlgebra:

    def test_invert_quaternion_cancels_rotation(self):
        for q in random_quats(30, seed=4):
            assert quat_close(quat_multiply(q, invert_quaternion(q)), IDENTITY_QUAT, atol=1e-10)

    def test_invert_quaternion_is_normalised(self):
        for q in random_quats(20, seed=5):
            assert np.isclose(np.linalg.norm(invert_quaternion(q)), 1.0)

    def test_quat_multiply_matches_matrix_product(self):
        q1s, q2s = random_quats(30, seed=6), random_quats(30, seed=7)
        for q1, q2 in zip(q1s, q2s, strict=True):
            assert np.allclose(
                quat2rot(quat_multiply(q1, q2)), quat2rot(q1) @ quat2rot(q2), atol=1e-10
            )

    def test_quat_error_is_zero_for_identical_rotations(self):
        for q in random_quats(20, seed=8):
            assert quat_error(q, q) == pytest.approx(0.0, abs=1e-9)

    def test_quat_error_is_bounded_by_pi(self):
        q1s, q2s = random_quats(50, seed=9), random_quats(50, seed=10)
        for q1, q2 in zip(q1s, q2s, strict=True):
            err = quat_error(q1, q2)
            assert 0.0 <= err <= np.pi + 1e-9

    def test_quat_error_recovers_a_known_angle(self):
        angle = 0.7
        q = Rotation.from_euler("z", angle).as_quat()
        assert quat_error(q, IDENTITY_QUAT) == pytest.approx(angle)


class TestWrapAngle:

    @pytest.mark.parametrize(
        "angle, expected",
        [(0.0, 0.0), (np.pi / 2, np.pi / 2), (-np.pi / 2, -np.pi / 2), (1.0, 1.0)],
    )
    def test_values_already_in_range_are_unchanged(self, angle, expected):
        assert wrap_angle(angle) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("angle", [3 * np.pi, -3 * np.pi, 2 * np.pi, 7.0, -7.0, 100.0])
    def test_wrapped_value_is_equivalent_modulo_two_pi(self, angle):
        """The sign at the +-pi boundary is a convention, so check equivalence, not a literal."""
        wrapped = float(wrap_angle(angle))
        assert abs(wrapped) <= np.pi + 1e-9
        # difference from the original must be an exact multiple of 2*pi
        k = (angle - wrapped) / (2 * np.pi)
        assert k == pytest.approx(round(k), abs=1e-9)

    def test_result_is_within_pi(self):
        angles = np.linspace(-20.0, 20.0, 401)
        wrapped = wrap_angle(angles)
        assert np.all(np.abs(wrapped) <= np.pi + 1e-9)

    def test_wrapping_preserves_the_rotation(self):
        angles = np.linspace(-20.0, 20.0, 101)
        for a in angles:
            q_raw = Rotation.from_euler("z", a).as_quat()
            q_wrapped = Rotation.from_euler("z", float(wrap_angle(a))).as_quat()
            assert quat_error(q_raw, q_wrapped) == pytest.approx(0.0, abs=1e-9)


class TestVecToSkew:

    def test_skew_is_antisymmetric(self):
        for v in random_positions(20, seed=11):
            skew = vec2skew(v)
            assert np.allclose(skew, -skew.T)

    def test_skew_reproduces_the_cross_product(self):
        rng = np.random.default_rng(12)
        for _ in range(20):
            a, b = rng.normal(size=3), rng.normal(size=3)
            assert np.allclose(vec2skew(a) @ b, np.cross(a, b))


class TestTransformationMatrices:

    def test_defaults_give_identity(self):
        assert np.allclose(transformation_matrix(), np.eye(4))

    def test_round_trip_through_pos_quat(self):
        for pos, quat in zip(random_positions(30, seed=13), random_quats(30, seed=14), strict=True):
            p, q = pos_quat_from_trans_mat(transformation_matrix(pos, quat))
            assert np.allclose(p, pos)
            assert quat_close(q, quat, atol=1e-10)

    def test_invert_transformation_matrix_matches_numpy_inverse(self):
        for pos, quat in zip(random_positions(30, seed=15), random_quats(30, seed=16), strict=True):
            mat = transformation_matrix(pos, quat)
            assert np.allclose(invert_transformation_matrix(mat), np.linalg.inv(mat), atol=1e-12)

    def test_transform_times_its_inverse_is_identity(self):
        for pos, quat in zip(random_positions(20, seed=17), random_quats(20, seed=18), strict=True):
            mat = transformation_matrix(pos, quat)
            assert np.allclose(mat @ invert_transformation_matrix(mat), np.eye(4), atol=1e-12)


class TestRigidTransformHelpers:
    """`invert_transform` / `multiply_transforms` replaced pybullet's `invertTransform` /
    `multiplyTransforms`, so they are checked against an independent float64 matrix reference."""

    def test_invert_transform_matches_matrix_reference(self):
        for pos, quat in zip(random_positions(50, seed=19), random_quats(50, seed=20), strict=True):
            ref_p, ref_q = pos_quat_from_trans_mat(np.linalg.inv(transformation_matrix(pos, quat)))
            got_p, got_q = invert_transform(pos, quat)
            assert np.allclose(got_p, ref_p, atol=1e-12)
            assert quat_close(got_q, ref_q, atol=1e-12)

    def test_multiply_transforms_matches_matrix_reference(self):
        pos_a, pos_b = random_positions(50, seed=21), random_positions(50, seed=22)
        quat_a, quat_b = random_quats(50, seed=23), random_quats(50, seed=24)
        for pa, qa, pb, qb in zip(pos_a, quat_a, pos_b, quat_b, strict=True):
            ref_p, ref_q = pos_quat_from_trans_mat(
                transformation_matrix(pa, qa) @ transformation_matrix(pb, qb)
            )
            got_p, got_q = multiply_transforms(pa, qa, pb, qb)
            assert np.allclose(got_p, ref_p, atol=1e-12)
            assert quat_close(got_q, ref_q, atol=1e-12)

    def test_transform_composed_with_its_inverse_is_identity(self):
        for pos, quat in zip(random_positions(30, seed=25), random_quats(30, seed=26), strict=True):
            p, q = multiply_transforms(*invert_transform(pos, quat), pos, quat)
            assert np.allclose(p, np.zeros(3), atol=1e-12)
            assert quat_close(q, IDENTITY_QUAT, atol=1e-10)

    def test_identity_is_a_no_op(self):
        for pos, quat in zip(random_positions(10, seed=27), random_quats(10, seed=28), strict=True):
            p, q = multiply_transforms(np.zeros(3), IDENTITY_QUAT, pos, quat)
            assert np.allclose(p, pos)
            assert quat_close(q, quat, atol=1e-12)

    def test_accepts_plain_sequences(self):
        p, q = multiply_transforms(
            [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        )
        assert np.allclose(p, [1.0, 2.0, 0.0])
        assert quat_close(q, IDENTITY_QUAT)
