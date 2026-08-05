"""Tests for `PinocchioInterface`.

These build a tiny URDF on the fly rather than downloading a robot description, so they are offline,
deterministic, and simple enough that mass/CoM/gravity values can be checked analytically.

Model (fixed base): two revolute joints about +y, links along +x, plus one continuous joint (which
exercises pinocchio's 2-value cos/sin representation that this interface hides).

    joint1 @ (0,0,0)  -> link1, mass 1, com at (0.5,0,0)
    joint2 @ (1,0,0)  -> link2, mass 1, com at (0.5,0,0) in link2  => (1.5,0,0) in world at q=0
    joint3 @ (1,0,0)  -> link3 (continuous), mass 0.5, com at the joint origin
"""

import numpy as np
import pytest

from pyrcf.utils.kinematics_dynamics import PinocchioInterface

G = 9.81

URDF = """<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0"/><mass value="0.0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="link2">
    <inertial>
      <origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="link3">
    <inertial>
      <origin xyz="0 0 0"/><mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="tip"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="1 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <joint name="joint3" type="continuous">
    <parent link="link2"/><child link="link3"/>
    <origin xyz="1 0 0"/><axis xyz="1 0 0"/>
    <limit effort="100" velocity="10"/>
  </joint>
  <joint name="tip_joint" type="fixed">
    <parent link="link3"/><child link="tip"/>
    <origin xyz="0 0 0"/>
  </joint>
</robot>
"""


def set_state(pin, joint_positions, joint_velocities=None, base_position=None):
    """`PinocchioInterface.update` requires the base arguments even for a fixed-base model."""
    n = len(pin.actuated_joint_names)
    pin.update(
        global_base_position=np.zeros(3) if base_position is None else np.asarray(base_position),
        global_base_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        local_base_velocity_linear=np.zeros(3),
        local_base_velocity_angular=np.zeros(3),
        joint_positions=np.asarray(joint_positions, float),
        joint_velocities=(
            np.zeros(n) if joint_velocities is None else np.asarray(joint_velocities, float)
        ),
        joint_order=pin.actuated_joint_names,
    )


@pytest.fixture(name="urdf_path")
def urdf_path_fixture(tmp_path):
    path = tmp_path / "test_arm.urdf"
    path.write_text(URDF)
    return str(path)


@pytest.fixture(name="pin")
def pin_fixture(urdf_path):
    iface = PinocchioInterface(urdf_filename=urdf_path, floating_base=False, verbose=False)
    set_state(iface, np.zeros(3))
    return iface


class TestModelStructure:

    def test_actuated_joints_are_discovered(self, pin):
        assert pin.actuated_joint_names == ["joint1", "joint2", "joint3"]
        assert pin.num_of_actuated_joints == 3

    def test_fixed_base_flag(self, pin):
        assert pin.floating_base is False

    def test_continuous_joint_is_identified(self, pin):
        """A continuous joint uses 2 values (cos, sin) in pinocchio's q vector."""
        assert len(pin.continuous_joint_q_ids) == 1

    def test_total_mass_is_the_sum_of_link_masses(self, pin):
        assert pin.get_centroidal_mass() == pytest.approx(2.5, abs=1e-9)

    def test_frame_lookup(self, pin):
        assert pin.get_frame_id("tip") >= 0


class TestKinematicsAtKnownConfigurations:

    def test_tip_position_at_zero_configuration(self, pin):
        pos, _ = pin.get_global_frame_pose("tip")
        assert np.allclose(pos, [2.0, 0.0, 0.0], atol=1e-9)

    def test_com_at_zero_configuration(self, pin):
        # 1 kg @ x=0.5, 1 kg @ x=1.5, 0.5 kg @ x=2.0  ->  com_x = 3.0 / 2.5 = 1.2
        assert np.allclose(pin.get_com_position(), [1.2, 0.0, 0.0], atol=1e-9)

    def test_tip_position_after_rotating_the_first_joint(self, pin):
        # +90 deg about +y maps +x onto -z
        set_state(pin, np.array([np.pi / 2, 0.0, 0.0]), np.zeros(3))
        pos, _ = pin.get_global_frame_pose("tip")
        assert np.allclose(pos, [0.0, 0.0, -2.0], atol=1e-6)

    def test_folding_the_second_joint(self, pin):
        set_state(pin, np.array([0.0, np.pi, 0.0]), np.zeros(3))
        pos, _ = pin.get_global_frame_pose("tip")
        assert np.allclose(pos, [0.0, 0.0, 0.0], atol=1e-6)

    def test_zero_velocity_gives_zero_frame_velocity(self, pin):
        assert np.allclose(pin.get_frame_velocity_linear("tip"), np.zeros(3), atol=1e-12)


class TestDynamics:

    def test_gravity_torques_match_the_analytic_values(self, pin):
        """At q=0 the arm lies along +x, so the gravity torque about joint1 is
        (1*0.5 + 1*1.5 + 0.5*2.0)*g and about joint2 (at x=1) is (1*0.5 + 0.5*1.0)*g."""
        g = pin.get_gravity_vector()
        assert abs(g[0]) == pytest.approx(3.0 * G, rel=1e-6)
        assert abs(g[1]) == pytest.approx(1.0 * G, rel=1e-6)

    def test_gravity_torque_vanishes_when_the_arm_hangs_vertically(self, pin):
        set_state(pin, np.array([np.pi / 2, 0.0, 0.0]), np.zeros(3))
        g = pin.get_gravity_vector()
        assert abs(g[0]) == pytest.approx(0.0, abs=1e-6)
        assert abs(g[1]) == pytest.approx(0.0, abs=1e-6)

    def test_gravity_about_the_continuous_joint_is_zero(self, pin):
        """joint3 rotates about +x and link3's com sits on that axis, so gravity exerts no torque
        about it regardless of link3's mass."""
        assert abs(pin.get_gravity_vector()[2]) == pytest.approx(0.0, abs=1e-9)

    def test_inertia_matrix_is_symmetric_positive_definite(self, pin):
        mass_matrix = pin.get_inertia_matrix()
        assert mass_matrix.shape[0] == mass_matrix.shape[1]
        assert np.allclose(mass_matrix, mass_matrix.T, atol=1e-9)
        assert np.all(np.linalg.eigvals(mass_matrix) > 0)

    def test_nonlinear_effects_equal_gravity_at_zero_velocity(self, pin):
        """With zero velocity the Coriolis/centrifugal term vanishes."""
        assert np.allclose(pin.get_nonlinear_effects(), pin.get_gravity_vector(), atol=1e-9)

    def test_nonlinear_effects_differ_from_gravity_when_moving(self, pin):
        set_state(pin, np.array([0.3, 0.4, 0.0]), np.array([1.5, -1.0, 0.0]))
        assert not np.allclose(pin.get_nonlinear_effects(), pin.get_gravity_vector(), atol=1e-6)


class TestJacobian:

    def test_linear_jacobian_matches_finite_differences(self, pin):
        q0 = np.array([0.2, -0.4, 0.0])
        set_state(pin, q0)
        jac = pin.get_frame_jacobian_linear("tip")

        eps = 1e-6
        numeric = np.zeros((3, 3))
        for i in range(3):
            q_plus, q_minus = q0.copy(), q0.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            set_state(pin, q_plus)
            plus, _ = pin.get_global_frame_pose("tip")
            set_state(pin, q_minus)
            minus, _ = pin.get_global_frame_pose("tip")
            numeric[:, i] = (plus.copy() - minus.copy()) / (2 * eps)

        # restore and compare the actuated-joint columns of the world-frame linear jacobian
        set_state(pin, q0)
        jac = pin.get_frame_jacobian_linear("tip")
        assert np.allclose(jac[:, -3:], numeric, atol=1e-4)

    def test_jacobian_shape(self, pin):
        assert pin.get_frame_jacobian("tip").shape[0] == 6

    def test_jacobian_is_zero_for_a_frame_on_the_base(self, pin):
        """No actuated joint moves the base link, so its jacobian must vanish."""
        assert np.allclose(pin.get_frame_jacobian_linear("base_link"), 0.0, atol=1e-12)


class TestContinuousJointHandling:

    def test_configuration_round_trip(self, pin):
        """A continuous joint is stored as (cos, sin); the interface must hide that."""
        q_joint = np.array([0.2, -0.4, 1.3])
        set_state(pin, q_joint, np.zeros(3))
        recovered = pin.get_joint_positions(pin.actuated_joint_names)
        assert np.allclose(recovered, q_joint, atol=1e-9)

    def test_continuous_joint_wraps(self, pin):
        """A continuous joint has no limits, so 2*pi past zero is the same configuration."""
        set_state(pin, np.array([0.0, 0.0, 2 * np.pi]), np.zeros(3))
        pos_wrapped, _ = pin.get_global_frame_pose("tip")
        set_state(pin, np.zeros(3), np.zeros(3))
        pos_zero, _ = pin.get_global_frame_pose("tip")
        assert np.allclose(pos_wrapped, pos_zero, atol=1e-6)

    def test_static_helpers_are_inverses(self):
        q_ids = [2]
        q = np.array([0.1, 0.2, 0.7, 0.0])
        encoded = PinocchioInterface.configuration_handle_continuous_joints(q.copy(), q_ids)
        # index 2 becomes cos(theta) and index 3 becomes sin(theta)
        assert encoded[2] == pytest.approx(np.cos(0.7))
        assert encoded[3] == pytest.approx(np.sin(0.7))
        decoded = PinocchioInterface.recover_generalised_pos_from_pinocchio_configuration(
            encoded, q_ids
        )
        assert decoded[2] == pytest.approx(0.7)

    def test_encoding_is_a_noop_without_continuous_joints(self):
        q = np.array([0.1, 0.2, 0.3])
        assert np.allclose(PinocchioInterface.configuration_handle_continuous_joints(q, []), q)


class TestFloatingBase:

    def test_floating_base_model_has_six_extra_dofs(self, urdf_path):
        fixed = PinocchioInterface(urdf_filename=urdf_path, floating_base=False, verbose=False)
        floating = PinocchioInterface(urdf_filename=urdf_path, floating_base=True, verbose=False)
        assert floating.model.nv == fixed.model.nv + 6
        assert floating.floating_base is True

    def test_floating_base_gravity_vector_includes_base_dofs(self, urdf_path):
        floating = PinocchioInterface(urdf_filename=urdf_path, floating_base=True, verbose=False)
        set_state(floating, np.zeros(3), base_position=np.zeros(3))
        g = floating.get_gravity_vector()
        assert len(g) == floating.model.nv
        # the vertical base dof carries the total weight
        assert abs(g[2]) == pytest.approx(2.5 * G, rel=1e-6)

    def test_floating_base_moves_the_com_with_the_base(self, urdf_path):
        floating = PinocchioInterface(urdf_filename=urdf_path, floating_base=True, verbose=False)
        set_state(floating, np.zeros(3), base_position=np.array([5.0, 0.0, 0.0]))
        assert np.allclose(floating.get_com_position(), [6.2, 0.0, 0.0], atol=1e-6)
