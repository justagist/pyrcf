from typing import List, Tuple, Callable, Any
from dataclasses import dataclass

from .ctrl_loop_debugger_base import CtrlLoopDebuggerBase
from ...utils.time_utils import ClockBase, PythonPerfClock
from ...core.types import GlobalMotionPlan, LocalMotionPlan, RobotState, RobotCmd
from ...core.types.debug_types import JointStatesCompareType, Pose3DCompareType
from ...utils.data_io_utils.pyrcf_publisher import PyRCFPublisherBase, PyRCFPublisherZMQ


@dataclass
class CtrlLoopDataStreamConfig:
    """Configurations for data streamer in the simple managed control loop.

    Get the configuration from the `SimpleManagedCtrlLoop` object using
    `control_loop.data_streamer_config`. Modify the parameters (see below) in this config as
    required before calling the `control_loop.run()` method.

    NOTE: Try to keep the rate of the publisher (stream_freq) a factor of the rate of the control
    loop. Otherwise, the RateTrigger will not be able to keep the required rate (because
    RateLimiter enforces a sleep).

    NOTE: Because the publisher uses a custom encoder for encoding PyRCF data types to serialisable
    data, this debugger can affect the speed of the control loop.

    Parameters:
        publish_robot_states (bool): Defaults to True.
        publish_agent_outputs (bool): Publish output from each agent in the control loop. (Local
            plan (if available) and RobotCmd from each agent). Defaults to True.
        publish_global_plan (bool): Defaults to True.
        publish_robot_cmd (bool): Defaults to True.
    """

    publish_robot_states: bool = True
    publish_global_plan: bool = True
    publish_agent_outputs: bool = True
    """Publish output from each agent in the control loop. (Local plan (if available)
    and RobotCmd from each agent). Defaults to True."""
    publish_robot_cmd: bool = True
    publish_joint_states_comparison: bool = True
    """If set to True, will publish joint states from different sources (robot state, 
    agent plan outputs, agent control command output, final robot command) in a 
    JointStatesCompareType object, to make it easier to compare values. (see
    `examples/utils_demo/demo_plotjuggler_loop_debugger.py` file)"""
    publish_ee_pose_comparison: bool = False
    """If set to True, will publish End-effector pose data from different sources
    (robot state (state estimator), agent plan outputs (if ee_reference is present)) 
    using Pose3DCompareType object, to make it easier to compare values. Only publishes
    poses for end-effectors available in `robot_state.state_estimates.ee_states.ee_names`.
    (See `examples/utils_demo/demo_plotjuggler_loop_debugger.py` file for details on
    how Pose3DCompareType can be used for other pose objects.)"""
    publish_debug_msg: bool = True
    """If set to True, will publish custom debug data from provided function handles."""
    prefix_name: str = "ctrl_loop"


class CtrlLoopDataPublisherBase(CtrlLoopDebuggerBase):
    """Base class for debuggers that stream the data from all the components in the control loop
    using some publisher that can stream strings/bytes. At the moment, supports only zmq and ros
    publishing.
    """

    def __init__(
        self,
        publisher: PyRCFPublisherBase = None,
        rate: float = None,
        clock: ClockBase = PythonPerfClock(),
        debug_publish_callables: Callable[[], Any] | List[Callable[[], Any]] = None,
    ):
        """Base class for debuggers that stream the data from all the components in the control
        loop to a specified tcp port using zmq, or via ros2 publisher topics.

        Args:
            port (int): tcp port to publish data to.
            rate (float, optional): Rate at which this should be triggered. Defaults to None
                (i.e. use control loop rate).
            clock (ClockBase, optional): The clock to use for timer. Defaults to PythonPerfClock().
            debug_publish_callables (Callable[[], Any] | List[Callable[[], Any]]): Function handle
                (or list of) that return publishable data (json encodable) for additional streaming
                to plotjuggler. Use this for debugging components. These method(s)/functions(s)
                will be called in the control loop and their return value added to the data being
                published (if publishing is enabled in self.data_streamer_config). Defaults to None.
            publisher_type (Literal["zmq", "ros"], optional): Type of publisher to use. Defaults to
                "zmq".
        """
        super().__init__(rate=rate, clock=clock)

        self._publish_data: bool = False
        self._publisher = publisher if publisher is not None else PyRCFPublisherZMQ()
        if rate is None or rate >= 0.0:
            self._publish_data = True

        self._additional_debug_handles = []
        self.add_debug_handle_to_publish(debug_publish_callables=debug_publish_callables)
        self._data_streamer_config = CtrlLoopDataStreamConfig()

    @property
    def data_streamer_config(self) -> CtrlLoopDataStreamConfig:
        """Config for the data stream published using this CtrlLoopDataPublisher debugger.

        Change its values before calling the run() method, if you want to modify them.

        Returns:
            CtrlLoopDataStreamConfig: Modifyable config for the data stream published
                using this CtrlLoopDataPublisher debugger.
        """
        return self._data_streamer_config

    def add_debug_handle_to_publish(
        self, debug_publish_callables: Callable[[], Any] | List[Callable[[], Any]]
    ):
        """Add additional function handles to publish extra debug data.

        See example 03.

        Args:
            debug_publish_callables (Callable[[], Any] | List[Callable[[], Any]]): Function handle
                (or list of) that return publishable data (json encodable) for additional streaming
                to plotjuggler. Use this for debugging components. These method(s)/functions(s)
                will be called in the control loop and their return value added to the data being
                published (if publishing is enabled in self.data_streamer_config).

        Raises:
            ValueError: If invalid value provided as argument.
        """
        if debug_publish_callables is not None:
            if callable(debug_publish_callables):
                debug_publish_callables = [debug_publish_callables]
            if not isinstance(debug_publish_callables, list):
                raise ValueError(
                    f"{self.__class__.__name__}: Argument to this function should"
                    " be a callable or a list of callables."
                )
            assert all(
                callable(ele) for ele in debug_publish_callables
            ), "All elements in `debug_publish_callables` should be a function handle."
            for handle in debug_publish_callables:
                if handle not in self._additional_debug_handles:
                    self._additional_debug_handles.append(handle)

    def _run_once_impl(
        self,
        t: float,
        dt: float,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        agent_outputs: List[Tuple[LocalMotionPlan, RobotCmd]],
        robot_cmd: RobotCmd,
    ):
        if not self._publish_data:
            return

        data = {"t": t, "dt": dt}
        if self._data_streamer_config.publish_robot_states:
            data["robot_states"] = robot_state
        if self._data_streamer_config.publish_global_plan:
            data["global_plan"] = global_plan
        if self._data_streamer_config.publish_agent_outputs:
            for n, agent_output in enumerate(agent_outputs):
                data[f"agent_{n+1}"] = {
                    "local_plan": agent_output[0],
                    "ctrl_cmd": agent_output[1],
                }
        if self._data_streamer_config.publish_robot_cmd:
            data["robot_cmd"] = robot_cmd
        if self._data_streamer_config.publish_joint_states_comparison:
            joint_states_compare = JointStatesCompareType(joint_states_list=[], row_names=[])
            # add joint state data from robot interface to compare
            joint_states_compare.joint_states_list.append(robot_state.joint_states)
            joint_states_compare.row_names.append("robot_state.joint_states")
            # add joint state reference from global planner
            joint_states_compare.joint_states_list.append(global_plan.joint_references)
            joint_states_compare.row_names.append("global_plan.joint_references")
            # add final robot joint commands to compare
            joint_states_compare.joint_states_list.append(robot_cmd.joint_commands)
            joint_states_compare.row_names.append("robot_cmd.joint_commands")
            # add joint commands from each agent separately to compare
            for n, agent_output in enumerate(agent_outputs):
                if agent_output[0] is not None:
                    joint_states_compare.joint_states_list.append(agent_output[0].joint_references)
                    joint_states_compare.row_names.append(
                        f"agent_{n+1}.local_plan.joint_references"
                    )
                joint_states_compare.joint_states_list.append(agent_output[1].joint_commands)
                joint_states_compare.row_names.append(f"agent_{n+1}.ctrl_cmd.joint_commands")
            data["joint_states_comparison"] = joint_states_compare
        if self._data_streamer_config.publish_ee_pose_comparison:
            ee_pose_compare = {}  # Pose3DCompareType(pose_list=[], row_names=[])
            if robot_state.state_estimates.end_effector_states.ee_poses is not None:
                ee_names = robot_state.state_estimates.end_effector_states.ee_names
                if ee_names is not None:
                    for ee_name in ee_names:
                        # ee pose from state estimate
                        ee_pose_compare[ee_name] = Pose3DCompareType(pose_list=[], row_names=[])
                        p, _, _, _ = robot_state.state_estimates.end_effector_states.get_state_of(
                            ee_name=ee_name
                        )
                        ee_pose_compare[ee_name].pose_list.append(p)
                        ee_pose_compare[ee_name].row_names.append("state_estimate")
                        # ee pose reference from global plan
                        try:
                            gpp, _, _, _ = global_plan.end_effector_references.get_state_of(
                                ee_name=ee_name
                            )
                        except (KeyError, AttributeError):
                            pass
                        else:
                            ee_pose_compare[ee_name].pose_list.append(gpp)
                            ee_pose_compare[ee_name].row_names.append(
                                "global_plan.end_effector_references"
                            )
                        # ee pose reference from local planner(s)/agents
                        for n, agent_output in enumerate(agent_outputs):
                            if agent_output[0] is not None:
                                try:
                                    pp, _, _, _ = agent_output[
                                        0
                                    ].end_effector_references.get_state_of(ee_name=ee_name)
                                except (KeyError, AttributeError):
                                    pass
                                else:
                                    ee_pose_compare[ee_name].pose_list.append(pp)
                                    ee_pose_compare[ee_name].row_names.append(
                                        f"agent_{n+1}.local_plan.end_effector_references"
                                    )
                    data["ee_pose_comparison"] = ee_pose_compare
        if self._data_streamer_config.publish_debug_msg:
            data["debug_data"] = [_get_data() for _get_data in self._additional_debug_handles]
        self._publisher.publish(data={self._data_streamer_config.prefix_name: data})

    def shutdown(self):
        self._publisher.close()
