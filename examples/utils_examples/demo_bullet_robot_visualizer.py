"""Demonstrating how to use PybulletRobotVisualizer to visualise a PybulletRobot instance to
for inspection/testing/debugging robot interfaces (states and joint values).

This visualizer exposes pybullet sliders to modify the joint positions and base pose of
the robot in the world.
"""

from pyrcf.components.robot_interfaces.simulation import PybulletRobot
from pyrcf.utils.sim_utils import PybulletRobotVisualizer

if __name__ == "__main__":

    # load any PybulletRobot instance here
    robot: PybulletRobot = PybulletRobot.fromAwesomeRobotDescriptions(
        robot_description_name="pepper_description",
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
        viz.run(sim_step_rate=240, slider_update_rate=10)
    except KeyboardInterrupt:
        viz.close()
