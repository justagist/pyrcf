"""Simple demo showing the use of some implmentations of pyrcf components joint position controller,
end-effector reference interpolator (local planner), and pybullet GUI for setting target
end-effector poses (implemented as a GlobalPlanner).

1. This demo shows how to load any robot from the Awesome robots list
(https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions)
as a PybulletRobot (RobotInterface derivative) instance.

2. This demo uses a pybullet GUI interface to set end-effector targets. This
is implemented as a `GlobalPlanner` (`UIBase`) called `PybulletGUIGlobalPlannerInterface`.

3. The local planner is `PybulletIKReferenceInterpolator` which uses a second order filter
to smoothly reach the joint positions for reaching the end-effector target set using
the GUI sliders.

4. The controller is a joint position (and velocity) tracking controller.

5. Also has an extended variant of this controller that adds gravity compensation torques to the
robot command, making the joint tracking much better.

6. There is also a debug robot in the simulation (using `PybulletRobotPlanDebugger`)
that shows the actual output from the planner (useful for controller tuning/debugging).
"""

from pyrcf.components.robot_interfaces.simulation import PybulletRobot
from pyrcf.control_loop import MinimalCtrlLoop
from pyrcf.components.local_planners import PybulletIKReferenceInterpolator
from pyrcf.components.global_planners.ui_reference_generators import (
    PybulletGUIGlobalPlannerInterface,
)
from pyrcf.components.controllers import JointPDController, GravityCompensatedPDController
from pyrcf.components.ctrl_loop_debuggers import PlotjugglerLoopDebugger, PybulletRobotPlanDebugger

# pylint: disable=W0212

USE_GRAVITY_COMP_CONTROLLER: bool = True
"""Set this flag to True to use a PD controller with gravity compensation.
If set to False, will use a regular PD joint position tracking controller.
Try enabling and disabling to see difference in tracking."""

if __name__ == "__main__":
    # load a manipulator robot from AwesomeRobotDescriptions (can load any robot
    # from the awesome robot lists using the `fromAwesomeRobotDescriptions` class
    # method.
    # https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions)
    robot: PybulletRobot = PybulletRobot.fromAwesomeRobotDescriptions(
        robot_description_name="iiwa14_description",
        floating_base=False,
        place_on_ground=False,
        # if `ee_names` is not defined here, will have to be provided as argument to
        # `PybulletGUIGlobalPlannerInterface` below. But setting it here is better as
        # it also adds FK for this end-effector in the state estimates for this
        # robot, which can be used as starting values for EE sliders
        ee_names=["iiwa_link_ee"],
    )

    state = robot.read()

    # create a "global planner" to provide end-effector targets using pybullet sliders
    global_planner = PybulletGUIGlobalPlannerInterface(
        enable_joint_sliders=False,
        cid=robot._sim_robot.cid,
        enable_ee_sliders=True,
        # ee_names=["iiwa_link_ee"], # not needed if defined in RobotInterface
    )

    # create a local planner that interpolates joint positions (using second-order
    # filter) to reach the end-effector targets from global planner (GUI sliders)
    local_planner = PybulletIKReferenceInterpolator(
        urdf_path=robot._sim_robot.urdf_path,
        floating_base=False,
        starting_base_position=state.state_estimates.pose.position,
        starting_base_orientation=state.state_estimates.pose.orientation,
        starting_joint_positions=state.joint_states.joint_positions,
        joint_names_order=state.joint_states.joint_names,
        filter_gain=0.03,
        blind_mode=True,
    )

    if USE_GRAVITY_COMP_CONTROLLER:
        # load a PD controller that adds gravity compensation torques to the
        # control command. This is just an extension of the naive
        # JointPositionVelocityController used below.
        controller = GravityCompensatedPDController(
            kp=[300, 300, 300, 300, 50, 50, 50],
            kd=[10, 8, 5, 0.1, 0.1, 0.1, 0.1],
            pinocchio_interface=robot.get_pinocchio_interface(),
        )
    else:
        # load position (and velocity) tracking controller without gravity compensation
        controller = JointPDController(
            kp=[300, 300, 300, 300, 50, 50, 50], kd=[10, 8, 5, 0.1, 0.1, 0.1, 0.1]
        )

    # create a control loop using these control loop components
    control_loop: MinimalCtrlLoop = MinimalCtrlLoop.useWithDefaults(
        robot_interface=robot,
        controller=controller,
        local_planner=local_planner,
        global_planner=global_planner,
    )

    ctrl_loop_rate: float = 240  # 240hz control loop

    try:
        plj = PlotjugglerLoopDebugger(rate=60)
        # enable ee pose comparison in the plotjuggler data stream
        plj.data_streamer_config.publish_ee_pose_comparison = True
        # this debugger can be used to visualise the planner output that the
        # controller is trying to track
        br_viz = PybulletRobotPlanDebugger(urdf_path=robot._sim_robot.urdf_path, rate=20)
        control_loop.run(
            loop_rate=ctrl_loop_rate,
            clock=robot.get_sim_clock(),
            debuggers=[plj, br_viz],
        )
    except KeyboardInterrupt:
        print("Closing control loop")
