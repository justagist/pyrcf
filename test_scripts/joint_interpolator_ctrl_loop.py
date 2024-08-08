"""Simple demo showing the use of awesome robots loader, joint position controller
(PD control), PD controller with gravity compensation, joint reference interpolator
(local planner), and pybullet GUI for setting target joint positions (implemented
as a GlobalPlanner).

1. This demo shows how to load any robot from the Awesome robots list
(https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions)
as a PybulletRobot (RobotInterface derivative) instance.

2. This demo uses a pybullet GUI interface to set joint position targets. This
is implemented as a `GlobalPlanner` (`UIBase`) called `PybulletGUIGlobalPlannerInterface`.

3. The local planner is `JointReferenceInterpolator` which uses a second order filter
to smoothly reach the joint targets from the GUI sliders.

4. The controller is a joint position (and velocity) tracking controller.

5. Extension of this controller that also adds gravity compensation torques to the
robot command, making the joint tracking much better.

6. There is also a debug robot in the simulation (using `PybulletRobotPlanDebugger`)
that shows the actual output from the planner (useful for controller tuning/debugging).
"""

from pyrcf.components.robot_interfaces.simulation import PybulletRobot
from pyrcf.control_loop import MinimalCtrlLoop
from pyrcf.components.local_planners import JointReferenceInterpolator
from pyrcf.components.global_planners.ui_reference_generators import (
    PybulletGUIGlobalPlannerInterface,
)
from pyrcf.components.controllers import JointPDController, GravityCompensatedPDController
from pyrcf.components.ctrl_loop_debuggers import (
    PlotjugglerLoopDebugger,
    PybulletRobotPlanDebugger,
)

# pylint: disable=W0212

USE_GRAVITY_COMP_CONTROLLER: bool = False
"""Set this flag to True to use a PD controller with gravity compensation.
If set to False, will use a regular PD joint position tracking controller.
Try enabling and disabling to see difference in tracking."""

SHOW_PLANNED_PATH: bool = True
"""Flag to enable/disable debug robot in pybullet that shows the local plan output."""

if __name__ == "__main__":
    # load a manipulator robot from AwesomeRobotDescriptions (can load any robot
    # from the awesome robot lists using the `fromAwesomeRobotDescriptions` class
    # method.
    # https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions)
    robot: PybulletRobot = PybulletRobot.fromAwesomeRobotDescriptions(
        robot_description_name="ur5_description",
        floating_base=False,
        place_on_ground=False,
    )

    # create a "global planner" to provide joint targets using pybullet sliders
    global_planner = PybulletGUIGlobalPlannerInterface(
        enable_joint_sliders=True,
        joint_lims=robot.get_pinocchio_interface().actuated_joint_limits,
        cid=robot._sim_robot.cid,
        # if urdf for the robot is provided (optional), the targets from the sliders
        # can be visualised
        js_viz_robot_urdf=robot._sim_robot.urdf_path,
    )

    # create a local planner that interpolates joint target from global planner
    # using second-order filter
    local_planner = JointReferenceInterpolator(filter_gain=0.03, blind_mode=True)

    if USE_GRAVITY_COMP_CONTROLLER:
        # load a PD controller that adds gravity compensation torques to the
        # control command. This is just an extension of the naive
        # JointPDController used below.
        controller = GravityCompensatedPDController(
            kp=[500, 500, 300, 5, 5, 5],
            kd=[10, 8, 5, 0.1, 0.1, 0.01],
            pinocchio_interface=robot.get_pinocchio_interface(),
        )
    else:
        # load position (and velocity) tracking controller
        controller = JointPDController(kp=[500, 500, 300, 5, 5, 5], kd=[10, 8, 5, 0.1, 0.1, 0.01])

    # create a control loop using these control loop components
    control_loop: MinimalCtrlLoop = MinimalCtrlLoop.useWithDefaults(
        robot_interface=robot,
        controller=controller,
        local_planner=local_planner,
        global_planner=global_planner,
    )

    ctrl_loop_rate: float = 240  # 240hz control loop
    debuggers = [PlotjugglerLoopDebugger(rate=None)]
    if SHOW_PLANNED_PATH:
        debuggers.append(
            # this debugger can be used to visualise the planner output that the
            # controller is trying to track
            PybulletRobotPlanDebugger(urdf_path=robot._sim_robot.urdf_path, rate=30)
        )
    try:
        control_loop.run(
            loop_rate=ctrl_loop_rate,
            clock=robot.get_sim_clock(),
            # PlotjugglerLoopDebugger will publish all data from control loop
            # to plotjuggler
            debuggers=debuggers,
        )
    except KeyboardInterrupt:
        print("Closing control loop")
