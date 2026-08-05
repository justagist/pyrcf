"""Offline contract tests for the MuJoCo robot interface.

These deliberately do NOT start a simulator or download robot descriptions, so they stay fast and
work without a network or a display. Behavioural verification of the backend lives in
`examples/utils_demos/demo_sim_backend_swap.py`.
"""

import inspect

import pytest

from pyrcf.components.robot_interfaces.robot_interface_base import RobotInterface
from pyrcf.components.robot_interfaces.simulation import (
    MujocoRobot,
    PybulletRobot,
    SimulatedRobotInterface,
)
from pyrcf.utils.time_utils import ClockBase


class TestMujocoRobotContract:

    def test_is_a_simulated_robot_interface(self):
        assert issubclass(MujocoRobot, SimulatedRobotInterface)
        assert issubclass(MujocoRobot, RobotInterface)

    def test_implements_every_abstract_method(self):
        """A missing abstract method would only surface at instantiation time."""
        assert not getattr(MujocoRobot, "__abstractmethods__", set())

    def test_exposes_the_same_core_api_as_the_pybullet_interface(self):
        """The two backends must be swappable, so the public surface has to line up."""
        for method in ["read", "write", "set_base_pose", "get_sim_clock", "shutdown"]:
            assert callable(getattr(MujocoRobot, method)), method
            assert callable(getattr(PybulletRobot, method)), method

    def test_world_clock_is_a_clockbase_and_advances_by_dt(self):
        clock = MujocoRobot.MujocoWorldClock(dt=0.002)
        assert isinstance(clock, ClockBase)
        assert clock.get_time() == 0.0
        clock.step_time()
        clock.step_time()
        assert clock.get_time() == pytest.approx(0.004)

    def test_from_awesome_robot_descriptions_accepts_a_pinocchio_urdf(self):
        """pinocchio cannot parse MJCF, so the loader must allow a separate URDF to be named."""
        params = inspect.signature(MujocoRobot.fromAwesomeRobotDescriptions).parameters
        assert "pinocchio_urdf_description" in params
        assert params["pinocchio_urdf_description"].default is None

    def test_constructor_accepts_both_model_formats(self):
        params = inspect.signature(MujocoRobot.__init__).parameters
        assert "urdf_path" in params and "mjcf_path" in params
        assert params["urdf_path"].default is None
        assert params["mjcf_path"].default is None
        # render must be switchable off for headless/CI use
        assert "render" in params

    def test_no_shared_mutable_defaults(self):
        """Same class of bug that was swept out of the rest of the package."""
        for func in [MujocoRobot.__init__, MujocoRobot.fromAwesomeRobotDescriptions]:
            for name, param in inspect.signature(func).parameters.items():
                if param.default is inspect.Parameter.empty:
                    continue
                assert not isinstance(
                    param.default, (list, dict, set)
                ), f"{func.__name__} has a mutable default for '{name}'"

    def test_requires_a_model_path(self):
        with pytest.raises((ValueError, RuntimeError), match="urdf_path|mjcf_path|mujoco-robot"):
            MujocoRobot()
