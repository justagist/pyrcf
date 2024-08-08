from typing import List, Tuple

from .ctrl_loop_debugger_base import CtrlLoopDebuggerBase
from ...core.types import GlobalMotionPlan, LocalMotionPlan, RobotCmd, RobotState
from ...utils.time_utils import ClockBase, PythonPerfClock
from ...utils.sim_utils.pybullet_debug_robot import PybulletDebugRobot
from ...components.callback_handlers.pb_gui_utils import PybulletGUIButton
from ...components.robot_interfaces.simulation.pybullet_robot import PybulletRobot


class PybulletRobotStateDebugger(CtrlLoopDebuggerBase):
    """This control loop debugger will create a dummy visual robot in pybullet which
    will simply mimic the joint states and base pose read from the `robot_state` (after
    passing through the state estimator update).
    This class can be used to visualise the robot states being read when the robot is
    running outside pybullet (e.g. real robot or remote).
    """

    def __init__(
        self,
        urdf_path: str,
        rate: float = None,
        clock: ClockBase = PythonPerfClock(),
        cid: int = 0,
        enable_toggle_button: bool = True,
        **bullet_debug_robot_kwargs,
    ):
        """This control loop debugger will create a dummy visual robot in pybullet which
        will simply mimic the joint states and base pose read from the `robot_state` (after
        passing through the state estimator update).
        This class can be used to visualise the robot states being read when the robot is
        running outside pybullet (e.g. real robot or remote).

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
            **bullet_debug_robot_kwargs: Additional keyword arguments can be passed here. All
                keyword arguments to `PybulletDebugRobot` is allowed here (e.g. use `rgba=[,,,]` for
                setting the visualisation color of this debug robot)
        """
        super().__init__(rate=rate, clock=clock)

        self._debugger_robot = PybulletDebugRobot(
            urdf_path=urdf_path, cid=cid, **bullet_debug_robot_kwargs
        )
        self._toggler_button = None
        if enable_toggle_button:
            self._toggler_button = PybulletGUIButton(name="Toggle robot state viz", cid=cid)

    def _run_once_impl(
        self,
        t: float,
        dt: float,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        agent_outputs: List[Tuple[LocalMotionPlan, RobotCmd]],
        robot_cmd: RobotCmd,
    ):
        self._debugger_robot.set_base_pose(
            position=robot_state.state_estimates.pose.position,
            orientation=robot_state.state_estimates.pose.orientation,
        )
        self._debugger_robot.set_joint_positions(
            joint_positions=robot_state.joint_states.joint_positions,
            joint_names=robot_state.joint_states.joint_names,
        )

    def shutdown(self):
        self._debugger_robot.close()

    @classmethod
    def fromBulletRobotInstance(
        cls: "PybulletRobotStateDebugger",
        robot: PybulletRobot,
        rate: float = None,
        clock: ClockBase = PythonPerfClock(),
        enable_toggle_button: bool = True,
        **bullet_debug_robot_kwargs,
    ) -> "PybulletRobotStateDebugger":
        """Creates an instance of PybulletRobotStateDebugger using an existing PybulletRobot
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
            **bullet_debug_robot_kwargs: Additional keyword arguments can be passed here. All
                keyword arguments to `PybulletDebugRobot` is allowed here (e.g. use `rgba=[,,,]` for
                setting the visualisation color of this debug robot)

        Returns:
            PybulletRobotStateDebugger: PybulletRobotStateDebugger instance created using the urdf and
                cid from the passed `robot` object.
        """
        assert isinstance(
            robot, PybulletRobot
        ), "The robot instance has to be derived from PybulletRobot base class."

        return cls(
            urdf_path=robot._sim_robot.urdf_path,  # pylint: disable=W0212
            rate=rate,
            clock=clock,
            cid=robot._sim_robot.cid,  # pylint: disable=W0212
            enable_toggle_button=enable_toggle_button,
            **bullet_debug_robot_kwargs,
        )
