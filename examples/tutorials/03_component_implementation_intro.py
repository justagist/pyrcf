"""Here we use actual implementations of control loop components instead of Dummy implementation,
and run a proper balancing controller on a simulated 2-wheeled segway-type robot (upkie). 

It also shows how a RobotInterface instance of any robot can be created in pybullet for
testing different controllers and planners.

Running this demo:
You can send velocity commands (forward, backward, rotate left, rotate right) to the robot
using keyboard (or joystick if connected) and make it move in the simulation.
"""

from pyrcf.components.controllers import SegwayPIDBalanceController
from pyrcf.components.robot_interfaces.simulation import PybulletRobot
from pyrcf.components.agents import PlannerControllerAgent
from pyrcf.components.local_planners import BlindForwardingPlanner
from pyrcf.components.state_estimators import DummyStateEstimator
from pyrcf.components.controller_manager import SimpleControllerManager
from pyrcf.components.global_planners.ui_reference_generators import (
    JoystickGlobalPlannerInterface,
    KeyboardGlobalPlannerInterface,
    DEFAULT_GAMEPAD_MAPPINGS,
    DEFAULT_KEYBOARD_MAPPING,
)
from pyrcf.control_loop import SimpleManagedCtrlLoop
from pyrcf.core.logging import logging
from pyrcf.core.exceptions import NotConnectedError

from pybullet_robot.utils.robot_loader_utils import get_urdf_from_awesome_robot_descriptions

logging.getLogger().setLevel(logging.DEBUG)


if __name__ == "__main__":
    # If using MinimalCtrlLoop, only requires defining a robot and a (compatible) controller.
    # (You can still add local_planner and global_planner if required.)

    # ** robot interface **
    # get robot urdf from robot descriptions loader. This function allows getting urdf of any robot from
    # the robot_descriptions package. See available robots here:
    # https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions
    upkie_urdf = get_urdf_from_awesome_robot_descriptions(
        robot_description_pkg_name="upkie_description"
    )
    # The PybulletRobot class can create a valid RobotInterface object that can be used in the
    # control loop, just by providing a urdf
    robot = PybulletRobot(
        urdf_path=upkie_urdf,
        floating_base=True,  # this robot does not have its base fixed in the world
        # the `ee_names` arg is used in creating a fully defined kinematics-dynamics
        # object (PinocchioInterface) for this robot (retrievable using the `get_pinocchio_interface()`
        # method of `PybulletRobot`). If `ee_names` is not provided, end-effector-based
        # kinematic funcitons of `PinocchioInterface` is not available, but other generic
        # kinematics dynamics functions can still be used.
        ee_names=["left_wheel_tire", "right_wheel_tire"],
    )
    # NOTE: the PybulletRobot has a helper classmethod to directly load a urdf from
    # the robot descriptions package as seen below:
    # ```
    # robot = PybulletRobot.fromAwesomeRobotDescriptions(
    #     robot_description_name="upkie_description",
    #     floating_base=True,  # this robot does not have its base fixed in the world
    #     ee_names=["left_wheel_tire", "right_wheel_tire"],
    # )
    # ```

    # ** state estimator **
    # all states are directly retrieved from the simulation in `PybulletRobot` class,
    # so we don't need a custom state estimator. So we just use the DummyStateEstimator.
    # Custom state estimators can be used here if needed.
    state_estimator = DummyStateEstimator(squawk=False)

    # ** global planner **
    # We use a user interface as a global planner to generate high-level commands to
    # the robot. Here, we use a Joystick interface if available, otherwise create
    # a keyboard interface (using pygame window)
    try:
        # use joystick if available (key bindings defined in DEFAULT_GAMEPAD_MAPPINGS
        # in `key_mappings.py`)
        global_planner = JoystickGlobalPlannerInterface(
            gamepad_mappings=DEFAULT_GAMEPAD_MAPPINGS, check_connection_at_init=True
        )
        logging.info("Using Joystick interface.")
    except (IndexError, NotConnectedError):
        # if joystick not available, use keyboard interface (pygame window should
        # be in focus for commands to work)
        logging.info("Could not detect joystick. Using Keyboard interface.")
        global_planner = KeyboardGlobalPlannerInterface(key_mappings=DEFAULT_KEYBOARD_MAPPING)

    # ** local planner **
    # In this example, we directly pass the velocity targets from the global planner (user
    # key input) and does not need a local planner to do anything. So we use the
    # `BlindForwardingPlanner` as the local planner in the control loop. This local
    # planner simply forwards the references from the GlobalMotionPlan into a
    # LocalMotionPlan object (which can then be used by a controller) directly without any
    # modification.
    local_planner = BlindForwardingPlanner()

    # ** controller **
    # create an instance of ControllerBase to be used in the control loop.
    # This controller is capable of balancing a 2-wheel segway type robot.
    # (Gains are tuned for this robot, task and simulator.)
    controller = SegwayPIDBalanceController(
        # the controller is also given access to the mutable `PinocchioInterface`
        # object of this robot. The `PinocchioInterface` object is automatically
        # updated for the real-time configuration of the robot in the simulation
        # in `PybulletRobot` (see `read()` method of `PybulletRobot` for
        # implementation).
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

    # ** agent **
    # wrap the local planner and controller into a control 'agent' as in previous example
    agent = PlannerControllerAgent(local_planner=local_planner, controller=controller)

    # ** controller manager **
    # create a controller manager to manage all the agents in the control loop
    controller_manager = SimpleControllerManager(agents=[agent])

    # use SimpleManagedCtrlLoop to run all the components
    control_loop = SimpleManagedCtrlLoop(
        robot_interface=robot,
        state_estimator=state_estimator,
        controller_manager=controller_manager,
        global_planner=global_planner,
    )

    ctrl_loop_rate: float = 200  # 200hz control loop
    # start control loop, kill it with Ctrl+C (keyboard interrupt is accepted by the control loop as
    # a shutdown signal, and the control loop will try to shut down all components gracefully)
    control_loop.run(loop_rate=ctrl_loop_rate)
