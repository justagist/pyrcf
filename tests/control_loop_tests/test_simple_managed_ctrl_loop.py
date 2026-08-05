"""Tests for control loop lifecycle guarantees (component shutdown on every exit path)."""

import pytest

from pyrcf.components.callback_handlers.base_callbacks import CustomCallbackBase
from pyrcf.components.controllers import ControllerBase
from pyrcf.components.ctrl_loop_debuggers.ctrl_loop_debugger_base import CtrlLoopDebuggerBase
from pyrcf.components.global_planners.ui_reference_generators.ui_base import DummyUI
from pyrcf.components.local_planners import DummyLocalPlanner
from pyrcf.components.robot_interfaces import DummyRobot
from pyrcf.components.state_estimators import DummyStateEstimator
from pyrcf.control_loop import MinimalCtrlLoop
from pyrcf.core.exceptions import CtrlLoopExitSignal
from pyrcf.core.types import RobotCmd


class RecordingRobot(DummyRobot):
    """DummyRobot that records whether it was deactivated and shut down."""

    def __init__(self):
        super().__init__(squawk=False)
        self.deactivated = False
        self.was_shutdown = False

    def deactivate(self) -> bool:
        self.deactivated = True
        return True

    def shutdown(self):
        self.was_shutdown = True


class RecordingDebugger(CtrlLoopDebuggerBase):
    """Debugger that records whether it was shut down (stands in for the data recorder, which
    must flush its buffer to disk on shutdown)."""

    def __init__(self):
        super().__init__(rate=None)
        self.was_shutdown = False

    def _run_once_impl(self, t, dt, robot_state, global_plan, agent_outputs, robot_cmd):
        return

    def shutdown(self):
        self.was_shutdown = True


class RecordingCallback(CustomCallbackBase):
    """Callback that records whether it was cleaned up."""

    def __init__(self):
        self.cleaned_up = False

    def run_once(self) -> None:
        return

    def cleanup(self) -> None:
        self.cleaned_up = True


class CountingController(ControllerBase):
    """Controller that raises the given exception after `fail_after` iterations."""

    def __init__(self, exception: BaseException = None, fail_after: int = 3):
        self.exception = exception
        self.fail_after = fail_after
        self.count = 0

    def update(self, robot_state, local_plan, t=None, dt=None) -> RobotCmd:
        self.count += 1
        if self.exception is not None and self.count >= self.fail_after:
            raise self.exception
        return RobotCmd()


def build_loop(controller: ControllerBase):
    robot = RecordingRobot()
    debugger = RecordingDebugger()
    callback = RecordingCallback()
    loop = MinimalCtrlLoop(
        robot_interface=robot,
        state_estimator=DummyStateEstimator(squawk=False),
        controller=controller,
        local_planner=DummyLocalPlanner(squawk=False),
        global_planner=DummyUI(squawk=False),
        verbose=False,
    )
    return loop, robot, debugger, callback


def assert_everything_cleaned_up(robot, debugger, callback):
    assert robot.deactivated, "robot was not deactivated"
    assert robot.was_shutdown, "robot was not shut down"
    assert debugger.was_shutdown, "debugger was not shut down (recorded data would be lost)"
    assert callback.cleaned_up, "callback was not cleaned up"


@pytest.mark.parametrize("signal", [KeyboardInterrupt, CtrlLoopExitSignal])
def test_components_shut_down_on_exit_signal(signal):
    loop, robot, debugger, callback = build_loop(CountingController(signal()))

    loop.run(loop_rate=500, debuggers=[debugger], poststep_callbacks=[callback])

    assert_everything_cleaned_up(robot, debugger, callback)


def test_components_shut_down_when_a_component_raises():
    """Regression test: a controller error (e.g. the segway controller's fall detection) used to
    escape the loop leaving the robot connected and recorded data unflushed."""
    error = RuntimeError("Base angle torso_pitch=1.2 rad denotes a fall")
    loop, robot, debugger, callback = build_loop(CountingController(error))

    with pytest.raises(RuntimeError, match="denotes a fall"):
        loop.run(loop_rate=500, debuggers=[debugger], poststep_callbacks=[callback])

    assert_everything_cleaned_up(robot, debugger, callback)


def test_shutdown_continues_when_one_component_fails():
    """A failure inside one shutdown step must not prevent the other components shutting down."""

    class BadDebugger(RecordingDebugger):
        def shutdown(self):
            raise RuntimeError("debugger shutdown failed")

    loop, robot, _, callback = build_loop(CountingController(CtrlLoopExitSignal()))
    bad_debugger = BadDebugger()

    loop.run(loop_rate=500, debuggers=[bad_debugger], poststep_callbacks=[callback])

    assert robot.deactivated
    assert robot.was_shutdown
    assert callback.cleaned_up, "callback cleanup was skipped because a debugger raised"


def test_loop_runs_and_tracks_iterations():
    loop, robot, debugger, callback = build_loop(
        CountingController(CtrlLoopExitSignal(), fail_after=10)
    )

    loop.run(loop_rate=500, debuggers=[debugger], prestep_callbacks=[callback])

    assert loop.get_loop_count() == 10
    true_dt, true_rate = loop.get_actual_loop_rate()
    assert true_dt > 0.0
    assert true_rate > 0.0
    assert_everything_cleaned_up(robot, debugger, callback)


class TestDefaultComponentsAreNotShared:
    """Regression tests: `useWithDefaults` used to evaluate its default planner and state
    estimator once at import time, so every control loop built through it shared one stateful
    instance of each."""

    def build_default_loop(self):
        return MinimalCtrlLoop.useWithDefaults(
            robot_interface=RecordingRobot(),
            controller=CountingController(),
            global_planner=DummyUI(squawk=False),
            verbose=False,
        )

    def agent_of(self, loop):
        (agent,) = loop.controller_manager.agents
        return agent

    def test_each_loop_gets_its_own_local_planner(self):
        loop_a, loop_b = self.build_default_loop(), self.build_default_loop()
        assert self.agent_of(loop_a).local_planner is not self.agent_of(loop_b).local_planner

    def test_each_loop_gets_its_own_state_estimator(self):
        loop_a, loop_b = self.build_default_loop(), self.build_default_loop()
        assert loop_a.state_estimator is not loop_b.state_estimator

    def test_each_loop_gets_its_own_controller_manager(self):
        loop_a, loop_b = self.build_default_loop(), self.build_default_loop()
        assert loop_a.controller_manager is not loop_b.controller_manager
        assert loop_a.controller_manager.agents is not loop_b.controller_manager.agents
