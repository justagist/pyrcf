"""Tests for `JointPDController` and `GravityCompensatedPDController`.

These run the controllers directly (no simulator) against synthetic `RobotState`/`LocalMotionPlan`
objects, so they are fast and offline. The gravity-compensated variant needs a `PinocchioInterface`,
which is faked (only `get_gravity_vector` and the actuated-joint index maps are used).
"""

import numpy as np
import pytest

from pyrcf.components.controllers import GravityCompensatedPDController, JointPDController
from pyrcf.core.types import (
    ControlMode,
    JointStates,
    LocalMotionPlan,
    RobotState,
)

N = 4
JOINT_NAMES = [f"j{i}" for i in range(N)]
KP = np.array([100.0, 200.0, 300.0, 400.0])
KD = np.array([1.0, 2.0, 3.0, 4.0])


def make_state(positions=None, velocities=None):
    return RobotState(
        joint_states=JointStates(
            joint_names=list(JOINT_NAMES),
            joint_positions=np.zeros(N) if positions is None else np.asarray(positions, float),
            joint_velocities=np.zeros(N) if velocities is None else np.asarray(velocities, float),
            joint_efforts=np.zeros(N),
        )
    )


def make_plan(mode=ControlMode.CONTROL, positions=None, velocities=None):
    plan = LocalMotionPlan(control_mode=mode)
    if positions is not None:
        plan.joint_references.joint_names = list(JOINT_NAMES)
        plan.joint_references.joint_positions = np.asarray(positions, float)
        plan.joint_references.joint_velocities = (
            np.zeros(N) if velocities is None else np.asarray(velocities, float)
        )
    return plan


class TestJointPDController:

    def test_tracks_the_reference_from_the_local_plan(self):
        ctrl = JointPDController(kp=KP, kd=KD)
        target = np.array([0.1, -0.2, 0.3, 0.4])
        cmd = ctrl.update(make_state(), make_plan(positions=target), t=0.0, dt=0.01)

        assert cmd.joint_commands.joint_names == JOINT_NAMES
        assert np.allclose(cmd.joint_commands.joint_positions, target)
        assert np.allclose(cmd.Kp, KP)
        assert np.allclose(cmd.Kd, KD)

    def test_gains_are_populated_on_the_first_call(self):
        ctrl = JointPDController(kp=KP, kd=KD)
        cmd = ctrl.update(make_state(), make_plan(positions=np.zeros(N)), t=0.0, dt=0.01)
        assert cmd.Kp is not None and cmd.Kd is not None
        assert len(cmd.Kp) == N and len(cmd.Kd) == N

    def test_scalar_gains_are_broadcast(self):
        ctrl = JointPDController(kp=50.0, kd=2.0)
        cmd = ctrl.update(make_state(), make_plan(positions=np.zeros(N)), t=0.0, dt=0.01)
        assert np.allclose(cmd.Kp, 50.0)
        assert np.allclose(cmd.Kd, 2.0)

    def test_idle_mode_with_hold_position_keeps_the_initial_target(self):
        ctrl = JointPDController(kp=KP, kd=KD, hold_position_at_start=True)
        start = np.array([0.5, 0.5, 0.5, 0.5])
        ctrl.update(make_state(positions=start), make_plan(mode=ControlMode.IDLE), t=0.0, dt=0.01)
        # robot has since drifted; the command must still point at the original pose
        cmd = ctrl.update(
            make_state(positions=start + 0.3), make_plan(mode=ControlMode.IDLE), t=0.01, dt=0.01
        )
        assert np.allclose(cmd.joint_commands.joint_positions, start)

    def test_idle_mode_without_hold_position_follows_the_measured_state(self):
        ctrl = JointPDController(kp=KP, kd=KD, hold_position_at_start=False)
        ctrl.update(make_state(), make_plan(mode=ControlMode.IDLE), t=0.0, dt=0.01)
        drifted = np.array([0.2, 0.2, 0.2, 0.2])
        cmd = ctrl.update(
            make_state(positions=drifted), make_plan(mode=ControlMode.IDLE), t=0.01, dt=0.01
        )
        assert np.allclose(cmd.joint_commands.joint_positions, drifted)

    def test_control_mode_without_joint_names_leaves_the_command_untouched(self):
        ctrl = JointPDController(kp=KP, kd=KD)
        target = np.array([0.1, 0.1, 0.1, 0.1])
        ctrl.update(make_state(), make_plan(positions=target), t=0.0, dt=0.01)
        # a CONTROL plan carrying no joint references must not wipe the previous target
        cmd = ctrl.update(make_state(), make_plan(mode=ControlMode.CONTROL), t=0.01, dt=0.01)
        assert np.allclose(cmd.joint_commands.joint_positions, target)

    def test_invalid_control_mode_raises(self):
        ctrl = JointPDController(kp=KP, kd=KD)
        plan = make_plan(positions=np.zeros(N))
        plan.control_mode = "not-a-mode"
        with pytest.raises(RuntimeError, match="Invalid control_mode"):
            ctrl.update(make_state(), plan, t=0.0, dt=0.01)

    def test_none_plan_is_tolerated(self):
        ctrl = JointPDController(kp=KP, kd=KD)
        cmd = ctrl.update(make_state(), None, t=0.0, dt=0.01)
        assert cmd is not None
        assert len(cmd.Kp) == N


class TestGainRamping:

    def test_gains_start_low_and_reach_the_target(self):
        ctrl = JointPDController(
            kp=KP, kd=KD, gain_ramp_up_time=1.0, ramp_start_kp=0.0, ramp_start_kd=0.0
        )
        plan = make_plan(positions=np.zeros(N))

        first = ctrl.update(make_state(), plan, t=0.0, dt=0.01)
        assert np.all(first.Kp < KP), "gains should start below target while ramping"

        # run past the ramp duration
        for i in range(200):
            cmd = ctrl.update(make_state(), plan, t=0.01 * i, dt=0.01)
        assert np.allclose(cmd.Kp, KP)
        assert np.allclose(cmd.Kd, KD)

    def test_gains_increase_monotonically_during_the_ramp(self):
        ctrl = JointPDController(kp=KP, kd=KD, gain_ramp_up_time=1.0)
        plan = make_plan(positions=np.zeros(N))
        seen = []
        for i in range(100):
            seen.append(float(ctrl.update(make_state(), plan, t=0.01 * i, dt=0.01).Kp[0]))
        assert all(
            b >= a - 1e-12 for a, b in zip(seen, seen[1:], strict=False)
        ), "ramp must be monotonic"

    def test_no_ramp_by_default(self):
        ctrl = JointPDController(kp=KP, kd=KD)
        cmd = ctrl.update(make_state(), make_plan(positions=np.zeros(N)), t=0.0, dt=0.01)
        assert np.allclose(cmd.Kp, KP), "gains should be at target immediately with no ramp"


class FakePinocchioInterface:
    """Minimal stand-in: the gravity-compensated controller needs the generalised gravity vector,
    the actuated joint names, and `floating_base` (which decides the 6-dof index offset into the
    gravity vector)."""

    def __init__(self, joint_gravity_torques, floating_base=False):
        self.floating_base = floating_base
        self.actuated_joint_names = list(JOINT_NAMES)
        joint_g = np.asarray(joint_gravity_torques, float)
        # a floating-base model's generalised gravity vector starts with the 6 base dofs
        self._g = np.concatenate([np.full(6, 999.0), joint_g]) if floating_base else joint_g

    def get_gravity_vector(self):
        return self._g


class TestGravityCompensatedPDController:

    def test_adds_gravity_torques_as_feedforward_effort(self):
        g = np.array([1.0, -2.0, 3.0, 0.5])
        ctrl = GravityCompensatedPDController(
            kp=KP, kd=KD, pinocchio_interface=FakePinocchioInterface(g)
        )
        cmd = ctrl.update(make_state(), make_plan(positions=np.zeros(N)), t=0.0, dt=0.01)

        assert cmd.joint_commands.joint_efforts is not None
        assert np.allclose(cmd.joint_commands.joint_efforts, g)

    def test_zero_gravity_gives_zero_feedforward(self):
        ctrl = GravityCompensatedPDController(
            kp=KP, kd=KD, pinocchio_interface=FakePinocchioInterface(np.zeros(N))
        )
        cmd = ctrl.update(make_state(), make_plan(positions=np.zeros(N)), t=0.0, dt=0.01)
        assert np.allclose(cmd.joint_commands.joint_efforts, np.zeros(N))

    def test_floating_base_gravity_vector_is_offset_by_six_dofs(self):
        """For a floating base the first 6 entries are the base dofs and must be skipped."""
        joint_g = np.array([1.0, -2.0, 3.0, 0.5])
        ctrl = GravityCompensatedPDController(
            kp=KP, kd=KD, pinocchio_interface=FakePinocchioInterface(joint_g, floating_base=True)
        )
        cmd = ctrl.update(make_state(), make_plan(positions=np.zeros(N)), t=0.0, dt=0.01)
        assert np.allclose(
            cmd.joint_commands.joint_efforts, joint_g
        ), "base dofs leaked into the joint efforts"

    def test_toggle_compensation_zeroes_the_feedforward(self):
        g = np.array([1.0, -2.0, 3.0, 0.5])
        ctrl = GravityCompensatedPDController(
            kp=KP, kd=KD, pinocchio_interface=FakePinocchioInterface(g)
        )
        plan = make_plan(positions=np.zeros(N))
        assert np.allclose(
            ctrl.update(make_state(), plan, t=0.0, dt=0.01).joint_commands.joint_efforts, g
        )

        ctrl.toggle_compensation(False)
        assert np.allclose(
            ctrl.update(make_state(), plan, t=0.01, dt=0.01).joint_commands.joint_efforts,
            np.zeros(N),
        )

        ctrl.toggle_compensation()  # no argument flips it back on
        assert np.allclose(
            ctrl.update(make_state(), plan, t=0.02, dt=0.01).joint_commands.joint_efforts, g
        )

    def test_still_tracks_position_like_the_plain_pd_controller(self):
        target = np.array([0.1, 0.2, -0.3, 0.4])
        ctrl = GravityCompensatedPDController(
            kp=KP, kd=KD, pinocchio_interface=FakePinocchioInterface(np.ones(N))
        )
        cmd = ctrl.update(make_state(), make_plan(positions=target), t=0.0, dt=0.01)
        assert np.allclose(cmd.joint_commands.joint_positions, target)
        assert np.allclose(cmd.Kp, KP)
