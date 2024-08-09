"""Demonstrating the use of `MinimalCtrlLoop` class which is useful when there is only
one agent in the control loop. This runs the exact same components as in the previous
example, and is equivalent demo, but simpler to write.

The `MinimalCtrlLoop` class is implemented from the `SimpleManagedCtrlLoop` introduced in
previous example, but is simplified for systems with only one controller and local planner
(i.e. one agent) in the control loop. Internally, the `MinimalCtrlLoop` does the creation of
Agent and Controller Manager for you. It also has some helpful defaults for components.
It also automatically connects to joystick or keyboard (as global planner) as available.

This demo does exactly the same thing as the previous example, but a lot of things are
hidden within the `MinimalCtrlLoop` class' implemenation.

Running this demo:
You can send velocity commands (forward, backward, rotate left, rotate right) to the robot
using keyboard (or joystick if connected) and make it move in the simulation.
"""

from pyrcf.components.controllers import SegwayPIDBalanceController
from pyrcf.components.robot_interfaces.simulation import PybulletRobot
from pyrcf.control_loop import MinimalCtrlLoop
from pyrcf.core.logging import logging

logging.getLogger().setLevel(logging.DEBUG)


if __name__ == "__main__":
    # If using MinimalCtrlLoop, only requires defining a robot and a (compatible) controller.
    # (You can still add local_planner and global_planner if required.)

    # ** robot interface **
    robot = PybulletRobot.fromAwesomeRobotDescriptions(
        robot_description_name="upkie_description",
        floating_base=True,  # this robot does not have its base fixed in the world
        ee_names=["left_wheel_tire", "right_wheel_tire"],
    )

    # ** controller **
    controller = SegwayPIDBalanceController(
        pinocchio_interface=robot.get_pinocchio_interface(),
        gains=SegwayPIDBalanceController.Gains(
            pitch_damping=10.0,
            pitch_stiffness=50.0,
            footprint_position_damping=5.0,
            footprint_position_stiffness=2.0,
            joint_kp=200.0,
            joint_kd=2.0,
            turning_gain_scale=1.0,
        ),
        max_integral_error_velocity=12,
        joint_torque_limits=[-1.0, 1.0],
    )

    # create a control loop using MinimalCtrlLoop class.
    control_loop: MinimalCtrlLoop = MinimalCtrlLoop.useWithDefaults(
        robot_interface=robot, controller=controller
    )
    # NOTE: the `useWithDefaults` classmethod automatically uses
    # dummy state estimator and forwarding planner (unless specified), as well
    # as connects to joystick or keyboard UI automatically (unless a different
    # global planner is specified)

    """
    # the above line is equivalent to doing the following:

    control_loop = MinimalCtrlLoop(
       robot_interface=robot,
       controller=controller,
       state_estimator=DummyStateEstimator(squawk=False),
       local_planner=BlindForwardingPlanner(),
       global_planner=JoystickInterface()  # or KeyboardInterface()
    )
    """

    ctrl_loop_rate: float = 200  # 200hz control loop
    # start control loop, kill it with Ctrl+C (keyboard interrupt is accepted by the control loop as
    # a shutdown signal, and the control loop will try to shut down all components gracefully)
    control_loop.run(loop_rate=ctrl_loop_rate)
