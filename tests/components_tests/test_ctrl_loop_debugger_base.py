"""Tests for `CtrlLoopDebuggerBase` rate gating.

`is_due()` exists so the control loop can skip gathering (and deep-copying) loop data for
debuggers that will not run this iteration. It must agree with the trigger that `run_once` consumes,
without consuming it itself.
"""

import time


from pyrcf.components.ctrl_loop_debuggers.ctrl_loop_debugger_base import CtrlLoopDebuggerBase
from pyrcf.core.types import GlobalMotionPlan, RobotCmd, RobotState


class CountingDebugger(CtrlLoopDebuggerBase):
    def __init__(self, rate=None):
        super().__init__(rate=rate)
        self.runs = 0

    def _run_once_impl(self, t, dt, robot_state, global_plan, agent_outputs, robot_cmd):
        self.runs += 1


def call(debugger):
    debugger.run_once(
        t=0.0,
        dt=0.01,
        robot_state=RobotState(),
        global_plan=GlobalMotionPlan(),
        agent_outputs=[(None, RobotCmd())],
        robot_cmd=RobotCmd(),
    )


class TestIsDue:

    def test_rate_none_is_always_due(self):
        dbg = CountingDebugger(rate=None)
        for _ in range(5):
            assert dbg.is_due() is True
            call(dbg)
        assert dbg.runs == 5

    def test_non_positive_rate_is_never_due(self):
        dbg = CountingDebugger(rate=0.0)
        assert dbg.is_due() is False
        for _ in range(5):
            call(dbg)
        assert dbg.runs == 0

    def test_is_due_does_not_consume_the_trigger(self):
        """Repeated peeks must not change the outcome, otherwise the loop would starve debuggers."""
        dbg = CountingDebugger(rate=1000.0)
        time.sleep(0.005)
        assert dbg.is_due() is True
        assert dbg.is_due() is True
        assert dbg.is_due() is True
        call(dbg)
        assert dbg.runs == 1

    def test_is_due_agrees_with_run_once(self):
        """If is_due() says True, the immediately following run_once must actually run."""
        dbg = CountingDebugger(rate=200.0)
        ran_when_due = 0
        due_count = 0
        for _ in range(200):
            if dbg.is_due():
                due_count += 1
                before = dbg.runs
                call(dbg)
                if dbg.runs > before:
                    ran_when_due += 1
            time.sleep(0.001)
        assert due_count > 0
        assert ran_when_due == due_count, "is_due() reported True but run_once() skipped"

    def test_not_due_immediately_after_running(self):
        dbg = CountingDebugger(rate=10.0)
        time.sleep(0.11)
        assert dbg.is_due() is True
        call(dbg)
        assert dbg.is_due() is False, "trigger should have been consumed by run_once"

    def test_becomes_due_again_after_the_period(self):
        dbg = CountingDebugger(rate=100.0)
        call(dbg)
        time.sleep(0.02)
        assert dbg.is_due() is True

    def test_rate_limited_debugger_runs_less_often_than_it_is_called(self):
        dbg = CountingDebugger(rate=50.0)
        deadline = time.perf_counter() + 0.2
        calls = 0
        while time.perf_counter() < deadline:
            call(dbg)
            calls += 1
        assert calls > dbg.runs, "a 50 Hz debugger should not run on every call"
        assert dbg.runs >= 1


class TestRunOncePassesCopies:

    def test_implementation_receives_copies_not_the_originals(self):
        received = {}

        class Capturing(CtrlLoopDebuggerBase):
            def _run_once_impl(self, t, dt, robot_state, global_plan, agent_outputs, robot_cmd):
                received["state"] = robot_state
                received["cmd"] = robot_cmd

        state, cmd = RobotState(), RobotCmd()
        dbg = Capturing(rate=None)
        dbg.run_once(
            t=0.0,
            dt=0.01,
            robot_state=state,
            global_plan=GlobalMotionPlan(),
            agent_outputs=[],
            robot_cmd=cmd,
        )
        assert received["state"] is not state
        assert received["cmd"] is not cmd


class TestShutdown:

    def test_base_shutdown_is_a_noop(self):
        assert CountingDebugger(rate=None).shutdown() is None
