"""Visualise any robot urdf (or robot description package from robot_description.py) in pybullet.

Useful for finding out joint values and joint limits.

Usage:
    pyrcf-visualise-robot <robot> [--floating-base]

    Run `pyrcf-visualise-robot --help` for the full argument list.
"""

import argparse
from pathlib import Path

from pyrcf.components.robot_interfaces.simulation import PybulletRobot
from pyrcf.core.logging import logger
from pyrcf.utils.sim_utils import PybulletRobotVisualizer

DESCRIPTION = (
    "Visualise any robot urdf (or robot description package from robot_descriptions.py) in "
    "pybullet. Useful for finding out joint values and joint limits."
)

ROBOT_ARG_HELP = (
    "Path to a urdf file, or the name of a description package from the `robot_descriptions.py` "
    "repo that has a valid urdf file "
    "(https://github.com/robot-descriptions/robot_descriptions.py#descriptions), "
    "e.g. `pepper_description`."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this executable.

    Returns:
        argparse.ArgumentParser: the parser for `pyrcf-visualise-robot`.
    """
    parser = argparse.ArgumentParser(
        prog="pyrcf-visualise-robot",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  pyrcf-visualise-robot pepper_description\n"
            "  pyrcf-visualise-robot /path/to/robot.urdf --floating-base\n"
        ),
    )
    parser.add_argument("robot", help=ROBOT_ARG_HELP)
    # NOTE: the robot is loaded with a fixed base by default, which is what you almost always
    # want for inspecting joint ranges.
    parser.add_argument(
        "--floating-base",
        action="store_true",
        help="Load the robot with a free-floating base instead of a fixed base.",
    )
    parser.add_argument(
        "--sim-step-rate",
        type=float,
        default=240.0,
        help="Rate (hz) at which the simulation is stepped. Defaults to 240.",
    )
    parser.add_argument(
        "--slider-update-rate",
        type=float,
        default=10.0,
        help="Rate (hz) at which the GUI sliders are read. Defaults to 10.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    floating_base: bool = args.floating_base
    logger.info(f"Robot has fixed base: {not floating_base}\n")

    if Path(args.robot).is_file():
        robot = PybulletRobot(
            urdf_path=args.robot, floating_base=floating_base, enable_torque_mode=False
        )
    else:
        robot: PybulletRobot = PybulletRobot.fromAwesomeRobotDescriptions(
            robot_description_name=args.robot,
            floating_base=floating_base,
            enable_torque_mode=False,
        )

    # load the visualiser using this robot
    viz: PybulletRobotVisualizer = PybulletRobotVisualizer.fromBulletRobot(
        pb_robot=robot,
        # optionally give starting joint positions (dictionary: joint_name -> joint_position)
        starting_joint_positions={},
        # optional list of strings to avoid when parsing through joints (eg. "_wheels")
        ignore_joints_with_str=[],
    )

    try:
        viz.run(sim_step_rate=args.sim_step_rate, slider_update_rate=args.slider_update_rate)
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()


if __name__ == "__main__":
    main()
