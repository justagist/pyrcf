"""This control loop debugger will create a dummy visual robot in pybullet which
will try to follow position/pose plan in the control loop. This can be used to
visualise ideal robot poses/configurations and to compare performance of
controllers in tracking them.
"""

from typing import List, Tuple, Mapping

from .ctrl_loop_debugger_base import CtrlLoopDebuggerBase
from ...core.types import (
    GlobalMotionPlan,
    LocalMotionPlan,
    PlannerMode,
    ControlMode,
    RobotCmd,
    RobotState,
)
from ...utils.time_utils import ClockBase, PythonPerfClock
from ...utils.sim_utils.pybullet_debug_robot import PybulletDebugRobot
from ...components.callback_handlers.pb_gui_utils import PybulletGUIButton, PybulletDebugFrameViz
from ...utils.frame_transforms import PoseTrasfrom
from ...utils.math_utils import rpy2quat
from ...components.robot_interfaces.simulation.pybullet_robot import PybulletRobot

# pylint: disable=W0212


class PybulletRobotPlanDebugger(CtrlLoopDebuggerBase):
    """This control loop debugger will create a dummy visual robot in pybullet which
    will try to follow position/pose plan in the control loop. This can be used to
    visualise ideal robot poses/configurations and to compare performance of
    controllers in tracking them.
    """

    def __init__(
        self,
        urdf_path: str,
        rate: float = None,
        clock: ClockBase = PythonPerfClock(),
        cid: int = 0,
        enable_toggle_button: bool = True,
        show_ee_targets: bool = True,
        use_curr_pose_if_plan_unavailable: bool = True,
        **pybullet_debug_robot_kwargs,
    ):
        """This control loop debugger will create a dummy visual robot in pybullet which
        will try to follow position/pose plan in the control loop. This can be used to
        visualise ideal robot poses/configurations and to compare performance of
        controllers in tracking them.

        Args:
            urdf_path (str): Path to urdf for the debug robot.
            rate (float, optional): Rate at which this should be triggered. Defaults to None
                (i.e. use control loop rate).
            clock (ClockBase, optional): The clock to use for timer. Defaults to PythonPerfClock().
            cid (int, optional): The pybullet physics client ID to connect to. Defaults to 0.
            enable_toggle_button (bool, optional): If set to True, this will create a button in
                the pybullet GUI to enable/disable the VISUALISATION of this debugger. Defaults to
                True.
                NOTE: This only affects the visualisation! Does not disable the actual debugger.
                NOTE: There can be an overhead when the visualisation is toggled on and off.

            **pybullet_debug_robot_kwargs: Additional keyword arguments can be passed here. All
                keyword arguments to `PybulletDebugRobot` is allowed here (e.g. use `rgba=[,,,]`
                for setting the visualisation color of this debug robot)
        """
        super().__init__(rate=rate, clock=clock)

        self._debugger_robot = PybulletDebugRobot(
            urdf_path=urdf_path, cid=cid, **pybullet_debug_robot_kwargs
        )
        self._toggler_button = None
        if enable_toggle_button:
            self._toggler_button = PybulletGUIButton(name="Toggle plan debugger viz", cid=cid)

        self._show_ee_targets = show_ee_targets
        self._ee_frame_viz: Mapping[str, PybulletDebugFrameViz] = {}
        self._use_curr_pose = use_curr_pose_if_plan_unavailable

    def _run_once_impl(
        self,
        t: float,
        dt: float,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        agent_outputs: List[Tuple[LocalMotionPlan, RobotCmd]],
        robot_cmd: RobotCmd,
    ):
        if global_plan.planner_mode == PlannerMode.IDLE:
            self._debugger_robot.set_base_pose(
                position=robot_state.state_estimates.pose.position,
                orientation=robot_state.state_estimates.pose.orientation,
            )
            return
        if self._toggler_button is not None and self._toggler_button.was_pressed():
            self._debugger_robot.toggle_visualisation()
        target_base_pose = (
            (
                robot_state.state_estimates.pose.position,
                robot_state.state_estimates.pose.orientation,
            )
            if self._use_curr_pose
            else None
        )
        joint_targets, joint_names = None, None
        for output in agent_outputs:
            # for each agent
            local_plan = output[0]
            control_cmd = output[1]
            if local_plan is not None and local_plan.control_mode == ControlMode.IDLE:
                # local plan can be None (e.g. for RL agent), but if there is a local
                # plan and it is setting the control mode to IDLE, we don't look any further
                continue

            # first check joint position from the control command and store this to
            # use as the debug visualisers value, but will be overwritten if there is a
            # reference in the local plan
            if control_cmd.joint_commands.joint_positions is not None:
                joint_targets = control_cmd.joint_commands.joint_positions
                joint_names = control_cmd.joint_commands.joint_names

            if local_plan is not None:
                if local_plan.relative_pose is not None:
                    # store target base pose for the debug robot if available in the local plan msg
                    target_base_pose = PoseTrasfrom.teleop2world(
                        p_in_teleop=local_plan.relative_pose.position,
                        q_in_teleop=rpy2quat(local_plan.relative_pose.rpy),
                        base_pos_in_world=robot_state.state_estimates.pose.position,
                        base_ori_in_world=robot_state.state_estimates.pose.orientation,
                    )
                if local_plan.joint_references.joint_positions is not None:
                    # overwrite any previous joint reference value for the debugger robot if this
                    # local plan message has joint references
                    joint_targets = local_plan.joint_references.joint_positions
                    joint_names = local_plan.joint_references.joint_names

            if (
                self._show_ee_targets
                and local_plan.end_effector_references.ee_names is not None
                and local_plan.end_effector_references.ee_poses is not None
            ):
                # update ee target frame visualisation if there is end-effector references pose
                # in the local plan message
                for n, ee_name in enumerate(local_plan.end_effector_references.ee_names):
                    pos = local_plan.end_effector_references.ee_poses[n].position
                    ori = local_plan.end_effector_references.ee_poses[n].orientation
                    if ee_name not in self._ee_frame_viz:
                        self._ee_frame_viz[ee_name] = PybulletDebugFrameViz(
                            cid=self._debugger_robot._cid,
                            position=pos,
                            orientation=ori,
                        )
                    else:
                        self._ee_frame_viz[ee_name].update_frame_pose(position=pos, orientation=ori)
        if target_base_pose is not None:
            self._debugger_robot.set_base_pose(
                position=target_base_pose[0], orientation=target_base_pose[1]
            )
        if joint_targets is not None:
            self._debugger_robot.set_joint_positions(
                joint_positions=joint_targets, joint_names=joint_names
            )

    def shutdown(self):
        self._debugger_robot.close()

    @classmethod
    def fromBulletRobotInstance(
        cls: "PybulletRobotPlanDebugger",
        robot: PybulletRobot,
        rate: float = None,
        clock: ClockBase = PythonPerfClock(),
        enable_toggle_button: bool = True,
        show_ee_targets: bool = True,
        use_curr_pose_if_plan_unavailable: bool = True,
        **pybullet_debug_robot_kwargs,
    ) -> "PybulletRobotPlanDebugger":
        """Creates an instance of PybulletRobotPlanDebugger using the provided PybulletRobot
        child class object.

        Args:
            robot (PybulletRobot): The robot instance (will not be modified).
            rate (float, optional): Rate at which this should be triggered. Defaults to None
                (i.e. use control loop rate).
            clock (ClockBase, optional): The clock to use for timer. Defaults to PythonPerfClock().
            cid (int, optional): The pybullet physics client ID to connect to. Defaults to 0.
            enable_toggle_button (bool, optional): If set to True, this will create a button in
                the pybullet GUI to enable/disable the VISUALISATION of this debugger. Defaults to
                True.
                NOTE: This only affects the visualisation! Does not disable the actual debugger.
                NOTE: There can be an overhead when the visualisation is toggled on and off.
            **pybullet_debug_robot_kwargs: Additional keyword arguments can be passed here. All
                keyword arguments to `PybulletDebugRobot` is allowed here (e.g. use `rgba=[,,,]`
                for setting the visualisation color of this debug robot)

        Returns:
            PybulletRobotPlanDebugger: PybulletRobotPlanDebugger instance created using the urdf and
                cid from the passed `PybulletRobot` object.
        """
        assert isinstance(
            robot, PybulletRobot
        ), "The robot instance has to be derived from PybulletRobot base class."

        return cls(
            urdf_path=robot._sim_robot.urdf_path,
            rate=rate,
            clock=clock,
            cid=robot._sim_robot.cid,
            enable_toggle_button=enable_toggle_button,
            show_ee_targets=show_ee_targets,
            use_curr_pose_if_plan_unavailable=use_curr_pose_if_plan_unavailable,
            **pybullet_debug_robot_kwargs,
        )
