"""Demonstrating the functionalities of the `PlotjugglerLoopDebugger` which can
be used to inspect all data in a control loop, by visualising them in plotjuggler.

Control Loop Debuggers are components that can intercept and read data passed by
the components in the control loop. These components read data (will not modify)
and can be used for visualising/debugging etc. For instance, the PlotjugglerLoopDebugger
is an implementation of the CtrlLoopDebuggerBase class, and intercepts all data
to publish them so that the data can be visualised in plotjuggler.

NOTE: Plotjuggler should be installed separately. See official documentation here:
https://github.com/facontidavide/PlotJuggler?tab=readme-ov-file#installation.

`PlotjugglerLoopDebugger` is an implementation of `CtrlLoopDebuggerBase` class
and therefore can be used with the control loop classes directly.

Use plotjuggler to visualise data published by this debugger (ZMQ subscriber;
msg protocol=json).

It can also publish custom data if required.

Also, see `examples/controllers_demo/ik_interpolator_position_controller_manipulator.py`
for example on how to use multiple debuggers, and to see how to use a visual debugging
tool for debugging controllers/planners (`BulletRobotPlanDebugger`).
"""

from pyrcf.components.robot_interfaces.simulation import PybulletRobot
from pyrcf.control_loop import MinimalCtrlLoop
from pyrcf.components.controllers import SegwayPIDBalanceController
from pyrcf.components.ctrl_loop_debuggers import PlotjugglerLoopDebugger, CtrlLoopDebuggerBase
from pyrcf.core.types import JointStates, EndEffectorStates, Pose3D
from pyrcf.core.types.debug_types import JointStatesCompareType, Pose3DCompareType
from pyrcf.utils.time_utils import PythonPerfClock

if __name__ == "__main__":
    robot = PybulletRobot.fromAwesomeRobotDescriptions(
        robot_description_name="upkie_description",
        floating_base=True,  # this robot does not have its base fixed in the world
        ee_names=["left_wheel_tire", "right_wheel_tire"],
    )

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

    # Create a minimal control loop
    control_loop: MinimalCtrlLoop = MinimalCtrlLoop.useWithDefaults(
        robot_interface=robot, controller=controller
    )

    # Create a debugger that can be used in the control loop.
    # The PlotjugglerLoopDebugger publishes data from all components in the
    # control loop to port 9872 by default. Data from all components can then
    # be visualised in plotjuggler (ZMQ subscriber; msg protocol=json).
    # All debuggers can take as argument a `rate` and `clock` parameter which
    # determines the frequency at which the debugger's main method is run
    # in the control loop. By default, it uses the `PythonPerfClock()` (system)
    # clock and runs at the same frequency as the control loop (`rate=None`).
    plotjuggler_debugger: CtrlLoopDebuggerBase = PlotjugglerLoopDebugger(
        rate=100, clock=PythonPerfClock()
    )
    # NOTE: setting the rate to None will publish data at the rate of the control loop

    # Additional configuration parameters for this debugger can be modified as follows.
    # # NOTE: all these values are the default, so you don't have to set them unless you
    # # want to change them
    plotjuggler_debugger.data_streamer_config.prefix_name = "ctrl_loop"
    # The above is the prefix name under which all data will be grouped when viewing
    # in plotjuggler.
    plotjuggler_debugger.data_streamer_config.publish_port = 9872  # port to publish to
    plotjuggler_debugger.data_streamer_config.publish_robot_states = True
    plotjuggler_debugger.data_streamer_config.publish_global_plan = True
    plotjuggler_debugger.data_streamer_config.publish_agent_outputs = True
    plotjuggler_debugger.data_streamer_config.publish_robot_cmd = True

    # if you want to publish custom data in addition to the data from the
    # control loop, you can also provide handles to functions that returns
    # additional publishable data. E.g.
    import time
    import numpy as np

    start_time = time.time()

    # creating a dummy function for generating extra data to publish
    # NOTE: The custom encoder defined in this library allows for
    # a more sensible representation when publishing data using the
    # datatypes defined in `pyrcf.core.types`
    # module as well.
    def _get_debug_data() -> dict:
        t = time.time() - start_time
        data = {
            "useless_time_data": {
                "t": t,
                "cos": np.cos(t),
                "sin": np.sin(t),
                "floor": np.floor(np.cos(t)),
                "ceil": np.ceil(np.cos(t)),
            },
            "Demo joint states data": JointStates(
                joint_names=["some", "random", "names"],
                joint_positions=[0, 1, 2],
                joint_efforts=[1, 4, 2],
            ),
            # Publishing Pose3D objects automatically encodes orientations as
            # RPY angles in degrees in addition to quaternions as well for
            # easier debugging
            "Demo ee states data": EndEffectorStates(
                ee_names=["ee_1", "ee_2"],
                ee_poses=[
                    Pose3D(),
                    Pose3D(position=np.ones(3) * t, orientation=np.array([0, 1, 0, 0])),
                ],
            ),
        }
        # Comparing lists, JointStates and Pose3D objects from different sources is made
        # easier using the datatypes defined in `types.debug_types`. E.g.
        new_js_obj = JointStates(
            # use same joint names for all jointstates objects to make it possible
            # to plot them side-by-side using the JointStatesCompareType encoding
            joint_names=["some", "random", "names"],
            joint_positions=[t, np.sin(t), np.cos(t)],
            joint_efforts=[2 * t, np.sin(2 * t), np.cos(2 * t)],
        )

        # add a JointStatesCompareType datatype to make it easier to compare different JointStates
        # objects in plotjugler. Just pass a list of JointStates objects that are to be compared.
        data["compare_joint_states_data"] = JointStatesCompareType(
            joint_states_list=[new_js_obj, data["Demo joint states data"]],
            # optional names for the different sources
            row_names=["source 1", "source 2"],
        )

        new_pose_obj = Pose3D(position=np.array([0, 1, 2]) * t, orientation=np.array([1, 0, 0, 0]))

        # add a Pose3DCompareType datatype to make it easier to compare different JointStates
        # objects in plotjugler. Just pass a list of Pose3D objects that are to be compared.
        data["compare_pose_obj"] = Pose3DCompareType(
            pose_list=[new_pose_obj, *(data["Demo ee states data"].ee_poses)],
            # optional names for the different sources
            row_names=["pose obj 1", *(data["Demo ee states data"].ee_names)],
        )

        return data

    # add a handle to this external function to the control loop publisher
    plotjuggler_debugger.add_debug_handle_to_publish(debug_publish_callables=[_get_debug_data])
    # NOTE: This could also have been done during construction of PlotjugglerDebugger:
    # ```
    # plotjuggler_debugger = PlotjugglerLoopDebugger(debug_publish_callables=[_get_debug_data])
    # ```

    # Tell the publisher to also stream the additional data we configured above. This
    # is already set to True by default
    plotjuggler_debugger.data_streamer_config.publish_debug_msg = True

    # In addition to publishing the raw data from the components, the debugger also
    # can be used to compare the different JointStates and Pose3D objects from the
    # outputs of the control loop components.
    plotjuggler_debugger.data_streamer_config.publish_joint_states_comparison = True
    # The above flag will make it easier to compare joint states data given from
    # different sources (robot state, planner output, controller output etc.)

    plotjuggler_debugger.data_streamer_config.publish_ee_pose_comparison = False
    # If the above value is set to True, will publish end-effector poses from
    # different sources (robot state estimate, local planner references (if
    # available)). By default this is set to False. See
    # `examples/controllers_demo/ik_interpolator_position_controller_manipulator.py`
    # for an example where this flag is set to True and can be used for comparing
    # robot end-effector pose vs target end-effector pose.

    # The comparison flags above basically make use of the JointStatesComparisonType
    # and Pose3DComparisonType objects respectively, similar to the example shown
    # in the custom function `get_debug_data` defined above.

    ctrl_loop_rate: float = 200  # 200hz control loop

    # run the control loop allowing the debugger to intercept and read the data
    control_loop.run(loop_rate=ctrl_loop_rate, debuggers=[plotjuggler_debugger])

    # All data can now be visualised in plotjuggler (ZMQ subscriber; msg protocol=json).


# Also, see `examples/controllers_demo/ik_interpolator_position_controller_manipulator.py`
# for example on how to use multiple debuggers, and to see how to use a visual debugging
# tool for debugging controllers/planners (`BulletRobotPlanDebugger`).
