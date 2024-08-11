"""Demo showing how `CustomCallbackBase` objects can be used in the control loop to interactively
modify things in the control loop.
`CustomCallbackBase` is intended to be used for debugging/developing/tuning etc. and not for
a final version of a working control task.

This example demonstrates how joint gain tuning can be done for `GravityCompensatedPDController`
using the `prestep_callbacks` (or `poststep_callbacks`) argument in the control loop.


1. This demo shows how to load any robot from the Awesome robots list
(https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions)
as a BulletRobot (RobotInterface derivative) instance.

2. This demo uses a pybullet GUI interface to set joint position targets. This
is implemented as a `GlobalPlanner` (`UIBase`) called `PybulletGUIGlobalPlannerInterface`.

3. The local planner is `JointReferenceInterpolator` which uses a second order filter
to smoothly reach the joint targets from the GUI sliders.

4. The controller is a joint position (and velocity) tracking controller.

5. Extension of this controller that also adds gravity compensation torques to the
robot command, making the joint tracking much better.

6. There is also a debug robot in the simulation (using `BulletRobotPlanDebugger`)
that shows the actual output from the planner (useful for controller tuning/debugging).
but creates sliders that can dynamically change the stiffness and damping gains for each joint
in the controller, as well as a button to toggle gravity compensation control. See the controller
demo first to understand the controller and planner being used in this demo.

7. Primarily, this example is used to demonstrate the CustomCallbacks that can be
defined in this framework to run additional things in loop. Here, this demo shows
how to use existing utility tools to create GUI sliders that can dynamically
change the stiffness and damping gains for each joint in the controller, as well
as a button to toggle gravity compensation control. It also shows how to define
your own custom callbacks to run in the control loop.
"""

from typing import List, Callable
from functools import partial
from pyrcf.components.robot_interfaces.simulation import PybulletRobot
from pyrcf.components.controllers import GravityCompensatedPDController
from pyrcf.control_loop import MinimalCtrlLoop
from pyrcf.components.local_planners import JointReferenceInterpolator
from pyrcf.components.global_planners.ui_reference_generators import (
    PybulletGUIGlobalPlannerInterface,
)
from pyrcf.components.ctrl_loop_debuggers import PlotjugglerLoopDebugger, PybulletRobotPlanDebugger
from pyrcf.utils.gui_utils import TkinterWidgetMaster
from pyrcf.components.callback_handlers import (
    RateTriggeredMultiCallbacks,
    TkMultiGUISliderSingleCallback,
    PbGUIButtonCallback,
    TkGUIButtonCallback,
    CustomCallbackBase,
)
from pyrcf.utils.time_utils import RateTrigger

# pylint: disable=W0212

if __name__ == "__main__":
    # load a robot
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
        js_viz_robot_urdf=robot._sim_robot.urdf_path,
    )

    # create a local planner that interpolates joint target from global planner
    # using second-order filter
    local_planner = JointReferenceInterpolator(filter_gain=0.03, blind_mode=True)

    # the default/starting values for kp and kd for each joint of the robot
    default_kp = [500, 500, 300, 5, 5, 5]
    default_kd = [10, 8, 5, 0.1, 0.1, 0.01]

    # load a gravity-compensating PD controller
    controller = GravityCompensatedPDController(
        kp=default_kp,
        kd=default_kd,
        pinocchio_interface=robot.get_pinocchio_interface(),
    )
    # disable gravity compensation to start with. This can be turned on in
    # GUI with a button callback defined below.
    controller.toggle_compensation(enable=False)
    robot_start_state = robot.read()

    # create a control loop using these control loop components
    control_loop: MinimalCtrlLoop = MinimalCtrlLoop.useWithDefaults(
        robot_interface=robot,
        controller=controller,
        local_planner=local_planner,
        global_planner=global_planner,
    )

    #  ----- How to create custom callbacks ------
    class MyCustomCallbackClass(CustomCallbackBase):
        def __init__(self, rate: float, string_to_print: str):
            """The init method can be specific to your usecase and does not need
            to conform to any specific rule. Here this class will print somthing in the
            control loop every 1/rate seconds."""

            # RateTrigger can be used to run a specific code only every 1/rate seconds
            # even if it is called more frequently.
            self._rate_trigger = RateTrigger(rate)
            self._text = string_to_print

        # NOTE: This method HAS to be implemented for any implementation of CustomCallbackBase
        # to be complete.
        def run_once(self) -> None:
            if self._rate_trigger.triggered():
                print(self._text)

        # NOTE: This method can be overridden (optional) by any child of CustomCallbackBase if
        # custom cleaning up is required before shutting down the program.
        def cleanup(self) -> None:
            print(f"\n\nCleaning up ({self._text}).\n\n")

    # --------------------------------------------

    # create dummy instances of the custom callback class we just created
    pre_step_custom_callback = MyCustomCallbackClass(1, "\npre-step\n")
    post_step_custom_callback = MyCustomCallbackClass(1, "\npost-step\n")

    # There are already several custom callback utilities defined for using GUI buttons, sliders, etc
    # using pybullet as well as using Tk library. They can be used to do different actions as desired.
    # Here, we show how some of these can be used for tuning controllers on the fly.

    #  ------- GUI CALLBACK STUFF ----------

    # define the callbacks for the sliders that will be defined
    # This callback will read values from one set of sliders and change the
    # controller's kp values
    def _set_kp(vals: List[float]):
        if controller._robot_cmd is not None:
            controller._robot_cmd.set_joint_gains(
                joint_names=robot_start_state.joint_states.joint_names, Kp=vals
            )

    # This callback will read values from another set of sliders and change the
    # controller's kd values
    def _set_kd(vals: List[float]):
        if controller._robot_cmd is not None:
            controller._robot_cmd.set_joint_gains(
                joint_names=robot_start_state.joint_states.joint_names, Kd=vals
            )

    ## create a list of all required gui callback objects
    # create 2 TkInter windows (one for kp sliders, one for kd sliders)
    tk_window_kp = TkinterWidgetMaster(window_name="Kp")
    tk_window_kd = TkinterWidgetMaster(window_name="Kd")
    # The `TkMultiGUISliderSingleCallback` class creates multiple sliders
    # in a tk window and uses values from all of these in one callback
    # This one is defined for setting kp (stiffness) for each joint.
    kp_sliders = TkMultiGUISliderSingleCallback(
        slider_names=[f"kp_{jname}" for jname in robot_start_state.joint_states.joint_names],
        slider_lower_lims=[0] * (len(default_kp)),
        slider_upper_lims=[2000] * len(default_kp),
        slider_default_vals=default_kp,
        tk_master=tk_window_kp,  # use first tk window
        callback=_set_kp,
    )
    # This one is defined for setting kd (damping) for each joint.
    kd_sliders = TkMultiGUISliderSingleCallback(
        slider_names=[f"kd_{jname}" for jname in robot_start_state.joint_states.joint_names],
        slider_lower_lims=[0] * (len(default_kd)),
        slider_upper_lims=[20] * len(default_kd),
        slider_default_vals=default_kd,
        tk_master=tk_window_kd,  # the second tk window
        callback=_set_kd,
    )

    # Create a callback function for resetting the slider values to their defaults
    def _reset_slider_vals(
        sliders_cb_obj: TkMultiGUISliderSingleCallback,
        vals: List[float],
        setter_function: Callable[[List[float]], None],  # either _set_kp or _set_kd
    ):
        setter_function(vals)  # reset kp or kd using _set_kp or _set_kd
        for n, slider in enumerate(sliders_cb_obj.get_sliders()):
            slider.set_value(vals[n])  # reset slider value

    # Create buttons in the corresponding tk windows to reset the gains
    # NOTE: Warning: resetting gains can be dangerous on the real robot
    reset_kp_button = TkGUIButtonCallback(
        "Reset kp",
        callback=partial(
            _reset_slider_vals,
            sliders_cb_obj=kp_sliders,
            vals=default_kp,
            setter_function=_set_kp,
        ),
        tk_master=tk_window_kp,
    )
    reset_kd_button = TkGUIButtonCallback(
        "Reset kd",
        callback=partial(
            _reset_slider_vals,
            sliders_cb_obj=kd_sliders,
            vals=default_kd,
            setter_function=_set_kd,
        ),
        tk_master=tk_window_kd,
    )
    # create a button with callback to enable and disable gravity compensation
    # control. This button is created in the pybullet GUI (not in tkinter).
    toggle_gravity_compensation_button = PbGUIButtonCallback(
        button_name="Toggle gravity compensation control",
        # The `toggle_compensation` method of this controller immediately
        # switches between enabling and disabling gravity compensation!
        # There is no smoothing! Don't use with real robot.
        callback=controller.toggle_compensation,
        cid=robot._sim_robot.cid,
    )
    # Button to request clean shutdown of control loop
    shutdown_button = PbGUIButtonCallback(
        button_name="Stop control loop",
        callback=control_loop.send_shutdown_signal,
        cid=robot._sim_robot.cid,
    )

    # The `RateTriggeredMultiCallbacks` class is the CustomCallbackBase object that
    # we will use in the control loop. This class can run callbacks from multiple
    # `CustomCallbackBase` objects at the specified (max) frequency. In this case, it
    # runs all the CustomCallbackBase objects we defined above. NOTE: we could have used
    # all the specified gui objects from above directly in the control loop, but
    # using the `RateTriggeredMultiCallbacks` ensures that they are triggered
    # at the same rate (for convenience). Otherwise each one will be checked in
    # every control loop iteration which could be inefficient.
    sliders_cb = RateTriggeredMultiCallbacks(
        gui_callbacks=[
            kp_sliders,
            kd_sliders,
            reset_kp_button,
            reset_kd_button,
            toggle_gravity_compensation_button,
            shutdown_button,
        ],
        rate=10,  # Hz
    )

    #  ------- GUI CALLBACK STUFF END ----------

    ctrl_loop_rate: float = 240  # 240hz control loop
    debuggers = [
        PlotjugglerLoopDebugger(rate=None),
        PybulletRobotPlanDebugger(urdf_path=robot._sim_robot.urdf_path, rate=30),
    ]
    try:
        control_loop.run(
            loop_rate=ctrl_loop_rate,
            clock=robot.get_sim_clock(),
            debuggers=debuggers,
            # add all the custom CustomCallbackBase objects; `prestep_callbacks` run before any other
            # components in the control loop, while `poststep_callbacks` can be used to run
            # callbacks after all components are executed in each control loop step.
            prestep_callbacks=[sliders_cb, pre_step_custom_callback],
            poststep_callbacks=[post_step_custom_callback],
        )
    except KeyboardInterrupt:
        print("Closing control loop")
