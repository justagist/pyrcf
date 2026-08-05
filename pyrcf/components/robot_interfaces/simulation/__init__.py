"""Provides simulation interfaces for robot extending and respecting the RobotInterface class.
These can be used in the control loop without needing a real robot.
"""

from .sim_robot_base import SimulatedRobotInterface
from .pybullet_robot import PybulletRobot

# NOTE: always importable so that `MujocoRobot` can give a clear install hint when used
# without the optional `mujoco-robot` dependency (same pattern as TorchScriptAgentBase).
from .mujoco_robot import MujocoRobot
