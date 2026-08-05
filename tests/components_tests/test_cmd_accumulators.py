import numpy as np
import pytest

from pyrcf.components.controller_manager.command_accumulators.cmd_accumulation_policies import (
    CmdMuxer,
    SimpleCmdOverride,
)
from pyrcf.core.types import RobotCmd


def make_cmd(joint_names, kp):
    cmd = RobotCmd.createZeros(len(joint_names), joint_names)
    cmd.Kp = kp
    return cmd


class TestSimpleCmdOverride:

    def test_returns_last_command(self):
        cmd1 = make_cmd(["j1"], [1.0])
        cmd2 = make_cmd(["j2"], [2.0])
        assert SimpleCmdOverride().accumulate([cmd1, cmd2]) is cmd2

    def test_single_command(self):
        cmd = make_cmd(["j1"], [1.0])
        assert SimpleCmdOverride().accumulate([cmd]) is cmd


class TestCmdMuxer:

    def test_single_command_is_returned(self):
        cmd = make_cmd(["j1", "j2"], [1.0, 2.0])
        out = CmdMuxer().accumulate([cmd])

        assert out is not None
        assert out.joint_commands.joint_names == ["j1", "j2"]
        assert np.all(out.Kp == np.array([1.0, 2.0]))

    def test_combines_disjoint_commands(self):
        cmd1 = make_cmd(["j1", "j2"], [1.0, 2.0])
        cmd2 = make_cmd(["j3"], [3.0])

        out = CmdMuxer().accumulate([cmd1, cmd2])

        assert out.joint_commands.joint_names == ["j1", "j2", "j3"]
        assert np.all(out.Kp == np.array([1.0, 2.0, 3.0]))

    def test_does_not_modify_inputs(self):
        cmd1 = make_cmd(["j1"], [1.0])
        cmd2 = make_cmd(["j2"], [2.0])

        CmdMuxer().accumulate([cmd1, cmd2])

        assert cmd1.joint_commands.joint_names == ["j1"]
        assert cmd2.joint_commands.joint_names == ["j2"]

    def test_conflicting_joints_rejected_by_default(self):
        cmd1 = make_cmd(["j1", "j2"], [1.0, 2.0])
        cmd2 = make_cmd(["j2"], [20.0])

        with pytest.raises(AssertionError):
            CmdMuxer().accumulate([cmd1, cmd2])

    def test_conflicting_joints_allowed_when_requested(self):
        cmd1 = make_cmd(["j1", "j2"], [1.0, 2.0])
        cmd2 = make_cmd(["j2"], [20.0])

        out = CmdMuxer(allow_conflicting_interfaces=True).accumulate([cmd1, cmd2])

        assert out.joint_commands.joint_names == ["j1", "j2"]
        assert np.all(out.Kp == np.array([1.0, 20.0]))

    def test_accumulator_is_stateless_across_calls(self):
        """Regression test: the muxer used to cache its first result and return it forever."""
        muxer = CmdMuxer()

        first = muxer.accumulate([make_cmd(["j1"], [1.0]), make_cmd(["j2"], [2.0])])
        second = muxer.accumulate([make_cmd(["j1"], [10.0]), make_cmd(["j2"], [20.0])])

        assert np.all(first.Kp == np.array([1.0, 2.0]))
        assert np.all(second.Kp == np.array([10.0, 20.0]))
