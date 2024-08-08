"""A Pybullet GUI interface to set joint targets for any robot.
This can be used as a GlobalPlanner (follows the GlobalMotionPlanner protocol)."""

from typing import Mapping, Tuple, List
from functools import partial
import copy

import pybullet as pb
import numpy as np

from .ui_base import UIBase
from ....core.types import (
    GlobalMotionPlan,
    RobotState,
    PlannerMode,
    Pose3D,
    Vector3D,
    QuatType,
)
from ...callback_handlers.pb_gui_utils import PybulletGUIButton
from ...callback_handlers.pb_gui_callbacks import (
    PbMultiGUISliderSingleCallback,
    PbGUIButtonCallback,
    PbDebugFrameVizCallback,
)
from ...callback_handlers.base_callbacks import RateTriggeredMultiCallbacks
from ....utils.time_utils import PythonPerfClock, ClockBase
from ....core.logging import logging
from ....utils.math_utils import quat2rpy, rpy2quat
from ....utils.sim_utils.pybullet_debug_robot import PbDebugRobotWithJointCallback


class PybulletGUIGlobalPlannerInterface(UIBase):
    """A Pybullet GUI interface to set joint/end-effector targets for any robot.
    This can be used as a GlobalPlanner (follows the GlobalMotionPlanner definition)."""

    def __init__(
        self,
        enable_joint_sliders: bool = True,
        joint_lims: Mapping[str, Tuple[float, float]] = None,
        enable_ee_sliders: bool = False,
        ee_names: List[str] = None,
        xyz_workspace_range: Tuple[Vector3D, Vector3D] = (
            np.array([-1, -1, -1]),
            np.array([1, 1, 1]),
        ),
        cid: int = None,
        slider_read_rate: float = 10,
        slider_rate_clock: ClockBase = PythonPerfClock(),
        js_viz_robot_urdf: str = None,
        js_viz_robot_rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.2),
    ):
        """A Pybullet GUI interface to set joint/end-effector targets for any robot.
        This can be used as a GlobalPlanner (follows the GlobalMotionPlanner definition).

        Args:
            enable_joint_sliders (bool, optional): If set to True, will create joint reference
                sliders. Defaults to True.
            joint_lims (Mapping[str, Tuple[float, float]], optional): mapping from
                {'joint_name' -> [lower_lim, upper_lim]}. Has to be provided if
                `enable_joint_sliders` is True. Defaults to None.
            enable_ee_sliders (bool, optional): If set to True, will create sliders for
                end-effector pose references. NOTE: `ee_names` have to be provided, or
                end-effector states should be part of the `robot_state` coming into this
                planner's update method (recommended). Defaults to False.
            ee_names (List[str], optional): List of end-effectors whose reference pose
                sliders should be generated. (It is recommended to leave this empty and
                let the interface detect end-effectors directly from the robot_state msg.)
            xyz_workspace_range (Tuple[Vector3D, Vector3D], optional): The range for the
                sliders when using end-effector target setting. Only valid if
                `enable_ee_sliders=True`. Defaults to ((-1,-1-,1), (1,1,1)).
                NOTE: If in the first update call (`process_user_input`), state estimates
                from the `robot_state` gives end-effector pose data for these end-effectors,
                this range is applied to the position data obtained from these state
                estimates.
            cid (int, optional): Client ID for the pybullet physics server to connect to.
                Defaults to None (will create a new pybullet GUI instance).
            slider_read_rate (float, optional): Rate at which the sliders should be read
                from the pybullet GUI. Defaults to 10 Hz.
            slider_rate_clock (ClockBase, optional): The clock to use for the slider rate.
                Defaults to `PythonPerfClock()` (system clock).
            js_viz_robot_urdf (str, optional): Path to urdf of the robot, if the joint targets
                from the sliders are to be visualised. Defaults to None (no visualisation of
                joint slider targets).
            js_viz_robot_rgba (Tuple[float, float, float, float], optional): RGBA color code
                for the visualisation of the robot (in case appropriate robot urdf was provided
                in `js_viz_robot_urdf`). Defaults to (1.0, 0.0, 0.0, 0.2).
        """
        self.cid = pb.connect(pb.GUI) if cid is None else cid
        self._enable_joint_sliders = enable_joint_sliders
        self._output_plan = GlobalMotionPlan(planner_mode=PlannerMode.CUSTOM)

        self._joint_sliders_enabled = False
        if self._enable_joint_sliders:
            assert joint_lims is not None, "Provide joint limits to use joint sliders"
            self._jnames = []
            self._j_lower_lims = []
            self._j_upper_lims = []
            for jname in joint_lims.keys():
                self._jnames.append(jname)
                self._j_lower_lims.append(joint_lims[jname][0])
                self._j_upper_lims.append(joint_lims[jname][1])
            self._output_plan.joint_references.joint_names = self._jnames
            self._desired_joint_positions: np.ndarray = None
            self._set_joints_button: PybulletGUIButton = None

        self._enable_ee_sliders = enable_ee_sliders
        self._ee_names_identified = False
        self._xyz_workspace_range = [
            np.array(xyz_workspace_range[0]),
            np.array(xyz_workspace_range[1]),
        ]

        if self._enable_ee_sliders and ee_names is not None:
            self._setup_ee_slider_inits(ee_names=ee_names)

        assert (
            self._enable_ee_sliders or self._enable_joint_sliders
        ), "Either `enable_joint_sliders` or `enable_ee_sliders` should be set to True."

        self._latest_base_pose: Tuple[Vector3D, QuatType] = None

        # create a rate-triggered gui callbacks object which can run multiple
        # GUICallback objects' callbacks at the specified rate
        self._pb_gui_callbacks = RateTriggeredMultiCallbacks(
            gui_callbacks=[],
            rate=slider_read_rate,
            clock=slider_rate_clock,
        )

        # add debug robot to visualise the slider targets (if possible)
        if (
            self._enable_joint_sliders
            and js_viz_robot_urdf is not None  # urdf for the robot has to be provided
            and js_viz_robot_rgba is not None  # rgba should have meaningful value
            and js_viz_robot_rgba[3] > 0  # alpha should be positive
        ):

            def _get_j_refs():
                return self._desired_joint_positions

            def _get_b_pose():
                return self._latest_base_pose

            # create a debug robot visualiser that uses the slider positions to change
            # joint position of the visual robot entitity
            _debug_robot = PbDebugRobotWithJointCallback(
                urdf_path=js_viz_robot_urdf,
                cid=self.cid,
                joint_names=self._jnames,
                get_joint_positions_callback=_get_j_refs,
                get_base_pose_callback=_get_b_pose,
                rgba=js_viz_robot_rgba,
            )
            # add this to the rate-triggered callback list
            self._pb_gui_callbacks.add_callback(gui_callback=_debug_robot)
            # also add a button to enable/disable this visualisation
            self._pb_gui_callbacks.add_callback(
                gui_callback=PbGUIButtonCallback(
                    button_name="Enable/disable joint target viz",
                    callback=_debug_robot._viz_robot.toggle_visualisation,
                    cid=self.cid,
                )
            )

    def _setup_ee_slider_inits(self, ee_names: List[str]):
        assert (
            not self._ee_names_identified
        ), "This should only have happened once, but was called twice."
        assert ee_names is not None and len(ee_names) > 0, "End-effectors could not be found."
        self._ee_names = ee_names
        self._output_plan.end_effector_references.ee_names = self._ee_names
        self._output_plan.end_effector_references.ee_poses = [None] * len(ee_names)
        self._ee_sliders_enabled = [False] * len(self._ee_names)
        self._desired_ee_poses: List[Pose3D] = [None] * len(ee_names)
        self._set_ee_pose_buttons: List[PybulletGUIButton] = [None] * len(ee_names)
        self._ee_names_identified = True

    def _set_joint_targets_cb(self, vals: List[float]):
        self._desired_joint_positions = np.array(vals)

    def _set_ee_pose_target_cb(self, vals: List[float], ee_id: int):
        self._desired_ee_poses[ee_id] = Pose3D(
            position=vals[:3], orientation=rpy2quat(np.deg2rad(vals[3:6]))
        )

    def _get_ee_target_pos_ori_cb(self, ee_id: int):
        return (
            self._desired_ee_poses[ee_id].position,
            self._desired_ee_poses[ee_id].orientation,
        )

    def process_user_input(
        self, robot_state: RobotState, t: float = None, dt: float = None
    ) -> GlobalMotionPlan:
        # run all rate-triggered gui callbacks that are defined
        self._pb_gui_callbacks.run_once()

        if self._enable_joint_sliders:
            # create the joint sliders and set-joints button if they don't exist
            if not self._joint_sliders_enabled:
                des_vals = [
                    robot_state.joint_states.get_state_of(joint_name=jname)[0]
                    for jname in self._jnames
                ]
                self._desired_joint_positions = des_vals
                self._pb_gui_callbacks.add_callback(
                    gui_callback=PbMultiGUISliderSingleCallback(
                        slider_names=self._jnames,
                        slider_lower_lims=self._j_lower_lims,
                        slider_upper_lims=self._j_upper_lims,
                        slider_default_vals=des_vals,
                        callback=self._set_joint_targets_cb,
                        cid=self.cid,
                    )
                )
                self._set_joints_button = PybulletGUIButton(name="Set joint target", cid=self.cid)
                self._joint_sliders_enabled = True
            # if button was pressed, set the joint target in the output plan
            if self._set_joints_button.was_pressed():
                self._output_plan.joint_references.joint_positions = (
                    self._desired_joint_positions.copy()
                )
            self._latest_base_pose = (
                robot_state.state_estimates.pose.position,
                robot_state.state_estimates.pose.orientation,
            )

        if self._enable_ee_sliders:
            # if ee names were not already given to constructor, find ee names
            # from state estimate (ee_states)
            if not self._ee_names_identified:
                # raise error if ee state estimate is not available and ee names were
                # not specified during construction
                assert (
                    robot_state.state_estimates.end_effector_states.ee_names is not None
                ), "No end-effector names found in state estimates. Provide end-effector name as argument to constructor or this should be part of the robot_state data."
                self._setup_ee_slider_inits(
                    ee_names=robot_state.state_estimates.end_effector_states.ee_names
                )
            for n, ee_name in enumerate(self._ee_names):
                # for each ee name, if sliders don't exist, create them here
                if not self._ee_sliders_enabled[n]:
                    # use current ee pose from state estimates if available, otherwise use
                    # identity at origin
                    try:
                        des_pose, _, _, _ = (
                            robot_state.state_estimates.end_effector_states.get_state_of(
                                ee_name=ee_name
                            )
                        )
                    except (AttributeError, KeyError):
                        logging.warning(
                            f"Starting pose of end-effector '{ee_name}' not found in state estimate. Using zeros as starting pose instead."
                        )
                        des_pose = Pose3D()
                    des_rpy = np.rad2deg(quat2rpy(des_pose.orientation))
                    # add ee pose sliders to callback list
                    self._pb_gui_callbacks.add_callback(
                        gui_callback=PbMultiGUISliderSingleCallback(
                            slider_names=[
                                f"x ({ee_name})",
                                f"y ({ee_name})",
                                f"z ({ee_name})",
                                f"roll ({ee_name})",
                                f"pitch ({ee_name})",
                                f"yaw ({ee_name})",
                            ],
                            slider_lower_lims=(
                                des_pose.position + self._xyz_workspace_range[0]
                            ).tolist()
                            + [-180, -180, -180],
                            slider_upper_lims=(
                                des_pose.position + self._xyz_workspace_range[1]
                            ).tolist()
                            + [180, 180, 180],
                            slider_default_vals=des_pose.position.tolist() + des_rpy.tolist(),
                            callback=partial(self._set_ee_pose_target_cb, ee_id=n),
                            cid=self.cid,
                        )
                    )
                    # add set ee pose button for this ee
                    self._set_ee_pose_buttons[n] = PybulletGUIButton(
                        name=f"{ee_name}: Set target pose", cid=self.cid
                    )

                    # create frame viz that shows desired ee pose from the
                    # slider values, add it to list of callbacks
                    self._pb_gui_callbacks.add_callback(
                        gui_callback=PbDebugFrameVizCallback(
                            callback=partial(self._get_ee_target_pos_ori_cb, ee_id=n),
                            cid=self.cid,
                        )
                    )
                    self._ee_sliders_enabled[n] = True
                # if button was pressed, set the pose for this ee in the output plan
                if self._set_ee_pose_buttons[n].was_pressed():
                    self._output_plan.end_effector_references.ee_poses[n] = copy.deepcopy(
                        self._desired_ee_poses[n]
                    )

        return self._output_plan

    def shutdown(self):
        super().shutdown()
        self._pb_gui_callbacks.cleanup()
