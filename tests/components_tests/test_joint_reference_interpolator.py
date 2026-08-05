"""Tests for `JointReferenceInterpolator` (local planner)."""

import numpy as np

from pyrcf.components.local_planners import BlindForwardingPlanner, JointReferenceInterpolator
from pyrcf.core.types import (
    ControlMode,
    GlobalMotionPlan,
    JointStates,
    PlannerMode,
    RobotState,
    Twist,
)

N = 3
JOINT_NAMES = [f"j{i}" for i in range(N)]


def make_state(positions=None):
    return RobotState(
        joint_states=JointStates(
            joint_names=list(JOINT_NAMES),
            joint_positions=np.zeros(N) if positions is None else np.asarray(positions, float),
            joint_velocities=np.zeros(N),
            joint_efforts=np.zeros(N),
        )
    )


def make_global_plan(targets, mode=PlannerMode.CUSTOM):
    plan = GlobalMotionPlan(planner_mode=mode)
    if targets is not None:
        plan.joint_references.joint_names = list(JOINT_NAMES)
        plan.joint_references.joint_positions = np.asarray(targets, float)
    return plan


class TestJointReferenceInterpolator:

    def test_non_custom_planner_mode_yields_an_empty_plan(self):
        planner = JointReferenceInterpolator()
        out = planner.generate_local_plan(
            make_state(), make_global_plan([0.5] * N, mode=PlannerMode.IDLE), t=0.0, dt=0.01
        )
        assert out.joint_references.joint_positions is None

    def test_output_is_in_control_mode_and_carries_joint_names(self):
        planner = JointReferenceInterpolator()
        out = planner.generate_local_plan(make_state(), make_global_plan([0.5] * N), t=0.0, dt=0.01)
        assert out.control_mode == ControlMode.CONTROL
        assert out.joint_references.joint_names == JOINT_NAMES

    def test_first_output_starts_from_the_measured_state_not_the_target(self):
        start = np.array([0.1, 0.2, 0.3])
        planner = JointReferenceInterpolator(filter_gain=0.1)
        out = planner.generate_local_plan(
            make_state(start), make_global_plan([1.0] * N), t=0.0, dt=0.01
        )
        # a smoothing filter must not jump straight to the target
        assert not np.allclose(out.joint_references.joint_positions, 1.0)
        assert np.all(np.abs(out.joint_references.joint_positions - start) < 0.5)

    def test_converges_to_the_target(self):
        target = np.array([0.4, -0.3, 0.2])
        planner = JointReferenceInterpolator(filter_gain=0.3)
        gp = make_global_plan(target)
        state = make_state()
        for i in range(3000):
            out = planner.generate_local_plan(state, gp, t=0.01 * i, dt=0.01)
        assert np.allclose(out.joint_references.joint_positions, target, atol=1e-4)

    def test_approach_is_monotonic_towards_the_target(self):
        target = np.array([1.0, 1.0, 1.0])
        planner = JointReferenceInterpolator(filter_gain=0.2)
        gp = make_global_plan(target)
        state = make_state()
        seen = []
        for i in range(200):
            out = planner.generate_local_plan(state, gp, t=0.01 * i, dt=0.01)
            seen.append(float(out.joint_references.joint_positions[0]))
        assert all(
            b >= a - 1e-12 for a, b in zip(seen, seen[1:], strict=False)
        ), "interpolation must not overshoot back"
        assert seen[-1] <= 1.0 + 1e-9

    def test_velocity_is_the_position_delta_over_dt(self):
        planner = JointReferenceInterpolator(filter_gain=0.2)
        gp = make_global_plan([1.0] * N)
        state = make_state()
        dt = 0.01
        prev = None
        for i in range(5):
            out = planner.generate_local_plan(state, gp, t=dt * i, dt=dt)
            pos = np.array(out.joint_references.joint_positions)
            vel = np.array(out.joint_references.joint_velocities)
            if prev is not None:
                assert np.allclose(vel, (pos - prev) / dt, atol=1e-9)
            prev = pos.copy()

    def test_zero_dt_gives_zero_velocity_instead_of_dividing_by_zero(self):
        planner = JointReferenceInterpolator(filter_gain=0.2)
        out = planner.generate_local_plan(make_state(), make_global_plan([1.0] * N), t=0.0, dt=0.0)
        assert np.allclose(out.joint_references.joint_velocities, 0.0)

    def test_blind_mode_ignores_joint_state_feedback(self):
        """In blind mode the interpolation continues from its own previous target, so a disturbed
        robot state must not change the generated reference."""
        gp = make_global_plan([1.0] * N)
        blind = JointReferenceInterpolator(filter_gain=0.2, blind_mode=True)
        sighted = JointReferenceInterpolator(filter_gain=0.2, blind_mode=False)

        for i in range(5):
            blind.generate_local_plan(make_state(), gp, t=0.01 * i, dt=0.01)
            sighted.generate_local_plan(make_state(), gp, t=0.01 * i, dt=0.01)

        # feed a disturbed state to both
        disturbed = make_state(np.full(N, -5.0))
        blind_next = blind.generate_local_plan(disturbed, gp, t=0.1, dt=0.01)
        sighted_next = sighted.generate_local_plan(disturbed, gp, t=0.1, dt=0.01)

        assert np.all(blind_next.joint_references.joint_positions > 0.0), "blind mode used feedback"
        assert np.all(
            sighted_next.joint_references.joint_positions < 0.0
        ), "non-blind mode ignored feedback"

    def test_other_global_plan_fields_are_forwarded_when_requested(self):
        gp = make_global_plan([0.5] * N)
        gp.twist = Twist(linear=np.array([1.0, 2.0, 3.0]), angular=np.array([0.1, 0.2, 0.3]))
        planner = JointReferenceInterpolator(forward_other_global_plan_values=True)
        out = planner.generate_local_plan(make_state(), gp, t=0.0, dt=0.01)
        assert np.allclose(out.twist.linear, [1.0, 2.0, 3.0])

    def test_other_global_plan_fields_are_not_forwarded_when_disabled(self):
        gp = make_global_plan([0.5] * N)
        gp.twist = Twist(linear=np.array([1.0, 2.0, 3.0]))
        planner = JointReferenceInterpolator(forward_other_global_plan_values=False)
        out = planner.generate_local_plan(make_state(), gp, t=0.0, dt=0.01)
        assert not np.allclose(out.twist.linear, [1.0, 2.0, 3.0])

    def test_switching_out_of_custom_mode_resets_the_interpolation(self):
        planner = JointReferenceInterpolator(filter_gain=0.2)
        gp = make_global_plan([1.0] * N)
        for i in range(50):
            planner.generate_local_plan(make_state(), gp, t=0.01 * i, dt=0.01)

        planner.generate_local_plan(
            make_state(), make_global_plan(None, mode=PlannerMode.IDLE), t=0.6, dt=0.01
        )
        # after the reset the interpolation must start again from the measured state
        out = planner.generate_local_plan(make_state(), gp, t=0.7, dt=0.01)
        assert out.joint_references.joint_positions[0] < 0.2


class TestBlindForwardingPlanner:

    def test_forwards_the_global_plan_verbatim_in_control_mode(self):
        gp = make_global_plan([0.7] * N)
        gp.twist = Twist(linear=np.array([1.0, 0.0, 0.0]))
        out = BlindForwardingPlanner().generate_local_plan(make_state(), gp, t=0.0, dt=0.01)

        assert out.control_mode == ControlMode.CONTROL
        assert np.allclose(out.joint_references.joint_positions, 0.7)
        assert np.allclose(out.twist.linear, [1.0, 0.0, 0.0])

    def test_does_not_smooth_or_alter_the_targets(self):
        gp = make_global_plan([0.0, 1.0, -1.0])
        out = BlindForwardingPlanner().generate_local_plan(make_state(), gp, t=0.0, dt=0.01)
        assert np.allclose(out.joint_references.joint_positions, [0.0, 1.0, -1.0])
