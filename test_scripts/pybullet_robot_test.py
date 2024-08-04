from pyrcf.components.robot_interfaces.simulation import PybulletRobot
import time

if __name__ == "__main__":

    robot: PybulletRobot = PybulletRobot.fromAwesomeRobotDescriptions(
        robot_description_name="pepper_description", enable_torque_mode=False
    )

    while True:
        try:
            robot.read()
        except KeyboardInterrupt:
            break
        time.sleep(0.01)
    print("done")
