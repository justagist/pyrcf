"""Visualise any robot urdf (or robot description package from robot_description.py) in pybullet.

Useful for finding out joint values and joint limits.

Usage: 
    pyrcf-visualise-robot <description_name> [as_floating_base? (1 or 0)]
    Args:
        <description_name>: Provide robot description name as first argument
            The robot description name should be a valid path to urdf file or 
            name of a description package from
            the `robot_descriptions.py` repo that has a valid urdf file: 
            https://github.com/robot-descriptions/robot_descriptions.py?tab=readme-ov-file#descriptions.
            e.g. `pepper_description`.
        (Optional): second argument can be 1 or 0 to load the robot as
            a fixed base robot (1) or not (0).
"""

from pyrcf.components.robot_interfaces.simulation import PybulletRobot
from pyrcf.utils.sim_utils import PybulletRobotVisualizer
from pyrcf.core.logging import logging
import sys
from pathlib import Path

usage_string = (
    "\nVisualise any robot urdf (or robot description package from robot_description.py) in pybullet.\n"
    "Useful for finding out joint values and joint limits.\n"
    "\n\nUsage: \n\n"
    "pyrcf-visualise-robot <description_name> [as_floating_base? (1 or 0)]\n\n"
    "Args:\n"
    "\t <description_name>: Provide robot description name as first argument\n"
    "\t\tThe robot description name should be a valid path to urdf file or \n"
    "\t\tname of a description package from\n"
    "\t\t the `robot_descriptions.py` repo that has a valid urdf file: \n"
    "\t\thttps://github.com/robot-descriptions/robot_descriptions.py?tab=readme-ov-file#descriptions.\n"
    "\t\t e.g. `pepper_description`.\n"
    "\t (Optional): second argument can be 1 or 0 to load the robot as\n"
    "\t\ta fixed base robot (1) or not (0).\n"
)


def main():

    if len(sys.argv) < 2:
        print(usage_string)
        return

    fixed_base: bool = True

    if len(sys.argv) > 2:
        fixed_base = bool(int(sys.argv[2]))

    logging.info(f"Robot has fixed base: {fixed_base}\n")

    try:
        if Path(sys.argv[1]).is_file():
            robot = PybulletRobot(
                urdf_path=sys.argv[1], floating_base=not fixed_base, enable_torque_mode=False
            )
        else:
            robot: PybulletRobot = PybulletRobot.fromAwesomeRobotDescriptions(
                robot_description_name=sys.argv[1],
                floating_base=not fixed_base,
                enable_torque_mode=False,
            )
    except Exception:
        print(usage_string)
        raise

    # load the visualiser using this robot
    viz: PybulletRobotVisualizer = PybulletRobotVisualizer.fromBulletRobot(
        pb_robot=robot,
        # optionally give starting joint positions (dictionary: joint_name -> joint_position)
        starting_joint_positions={},
        # optional list of strings to avoid when parsing through joints (eg. "_wheels")
        ignore_joints_with_str=[],
    )

    try:
        viz.run(sim_step_rate=240, slider_update_rate=10)
    except KeyboardInterrupt:
        viz.close()
