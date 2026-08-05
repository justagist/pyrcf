"""Round-trip tests: `ComponentDataRecorderDebugger` writes, `ComponentDataRecorderDataParser`
reads it back.

The recorder buffers and pickles in batches, and flushes the remainder on shutdown, so the
interesting cases are buffer boundaries (exact multiple, partial final batch, nothing recorded).
"""

import numpy as np
import pytest

from pyrcf.components.ctrl_loop_debuggers import ComponentDataRecorderDebugger
from pyrcf.core.types import (
    GlobalMotionPlan,
    JointStates,
    LocalMotionPlan,
    RobotCmd,
    RobotState,
)
from pyrcf.utils.data_io_utils import ComponentDataRecorderDataParser

N = 3
JOINT_NAMES = [f"j{i}" for i in range(N)]


def make_state(step):
    return RobotState(
        joint_states=JointStates(
            joint_names=list(JOINT_NAMES),
            joint_positions=np.full(N, float(step)),
            joint_velocities=np.zeros(N),
            joint_efforts=np.zeros(N),
        )
    )


def record(path, n_steps, buffer_size=50, extra=None):
    dbg = ComponentDataRecorderDebugger(
        file_name=str(path), rate=None, buffer_size=buffer_size, extra_data_callables=extra
    )
    for step in range(n_steps):
        dbg.run_once(
            t=step * 0.01,
            dt=0.01,
            robot_state=make_state(step),
            global_plan=GlobalMotionPlan(),
            agent_outputs=[(LocalMotionPlan(), RobotCmd.createZeros(N, list(JOINT_NAMES)))],
            robot_cmd=RobotCmd.createZeros(N, list(JOINT_NAMES)),
        )
    dbg.shutdown()
    return dbg


class TestRoundTrip:

    @pytest.mark.parametrize("n_steps", [1, 7, 50, 120])
    def test_all_recorded_steps_are_read_back(self, tmp_path, n_steps):
        path = tmp_path / "rec.pkl"
        record(path, n_steps, buffer_size=50)

        parser = ComponentDataRecorderDataParser(file_name=str(path))
        assert parser.num_datapoints == n_steps

    def test_exact_buffer_multiple_is_not_truncated(self, tmp_path):
        """A batch boundary is where a flush-on-shutdown bug would hide."""
        path = tmp_path / "rec.pkl"
        record(path, 100, buffer_size=50)
        assert ComponentDataRecorderDataParser(file_name=str(path)).num_datapoints == 100

    def test_partial_final_batch_is_flushed_on_shutdown(self, tmp_path):
        path = tmp_path / "rec.pkl"
        record(path, 55, buffer_size=50)
        assert ComponentDataRecorderDataParser(file_name=str(path)).num_datapoints == 55

    def test_values_survive_the_round_trip_in_order(self, tmp_path):
        path = tmp_path / "rec.pkl"
        record(path, 30, buffer_size=8)

        data = ComponentDataRecorderDataParser(file_name=str(path)).get_all_data()
        assert [round(t, 6) for t in data["t"]] == [round(i * 0.01, 6) for i in range(30)]
        for step, state in enumerate(data["robot_state"]):
            assert np.allclose(state.joint_states.joint_positions, float(step))
            assert state.joint_states.joint_names == JOINT_NAMES

    def test_all_expected_keys_are_present(self, tmp_path):
        path = tmp_path / "rec.pkl"
        record(path, 5)
        parser = ComponentDataRecorderDataParser(file_name=str(path))
        assert set(parser.key_names) == {
            "t",
            "dt",
            "robot_state",
            "global_plan",
            "agent_outputs",
            "robot_cmd",
            "debug_data",
        }

    def test_recording_nothing_produces_an_empty_but_readable_file(self, tmp_path):
        path = tmp_path / "rec.pkl"
        record(path, 0)
        assert ComponentDataRecorderDataParser(file_name=str(path)).num_datapoints == 0

    def test_lazy_loading_defers_until_queried(self, tmp_path):
        path = tmp_path / "rec.pkl"
        record(path, 4)
        parser = ComponentDataRecorderDataParser(file_name=str(path), load_on_init=False)
        assert parser.num_datapoints is None
        assert len(parser.get_all_data()["t"]) == 4


class TestExtraDataCallables:

    def test_extra_callable_output_is_recorded(self, tmp_path):
        path = tmp_path / "rec.pkl"
        record(path, 6, extra=lambda: 42)

        data = ComponentDataRecorderDataParser(file_name=str(path)).get_all_data()
        assert all(d == [42] for d in data["debug_data"])

    def test_multiple_extra_callables(self, tmp_path):
        path = tmp_path / "rec.pkl"
        record(path, 4, extra=[lambda: 1, lambda: "two"])

        data = ComponentDataRecorderDataParser(file_name=str(path)).get_all_data()
        assert all(d == [1, "two"] for d in data["debug_data"])

    def test_non_callable_extra_data_is_rejected(self, tmp_path):
        with pytest.raises((ValueError, AssertionError)):
            ComponentDataRecorderDebugger(
                file_name=str(tmp_path / "x.pkl"), extra_data_callables=["not a callable"]
            )


class TestFieldQueries:

    def test_get_all_data_for_key_returns_that_key(self, tmp_path):
        path = tmp_path / "rec.pkl"
        record(path, 10)
        parser = ComponentDataRecorderDataParser(file_name=str(path))
        ts = parser.get_all_data_for_key("t")
        assert len(ts) == 10

    def test_nested_field_query_across_timesteps(self, tmp_path):
        path = tmp_path / "rec.pkl"
        record(path, 10)
        parser = ComponentDataRecorderDataParser(file_name=str(path))

        positions = parser.get_all_data_for_key("robot_state", "joint_states.joint_positions")
        positions = np.asarray(positions)
        assert positions.shape == (10, N)
        # the recorder wrote joint_positions = full(N, step) at each step
        for step in range(10):
            assert np.allclose(positions[step], float(step))

    def test_indexed_field_query(self, tmp_path):
        path = tmp_path / "rec.pkl"
        record(path, 6)
        parser = ComponentDataRecorderDataParser(file_name=str(path))
        first_joint = np.asarray(
            parser.get_all_data_for_key("robot_state", "joint_states.joint_positions[0]")
        )
        assert np.allclose(first_joint, np.arange(6, dtype=float))
