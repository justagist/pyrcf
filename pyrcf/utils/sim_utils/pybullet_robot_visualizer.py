from typing import Mapping, List, Dict, Callable
import threading
import copy
from functools import partial
import pybullet as pb
import numpy as np
import pybullet_robot

from ...components.robot_interfaces.simulation.pybullet_robot import PybulletRobot
from ..time_utils import RateLimiter
from ..math_utils import rpy2quat, quat2rpy
from ...core.logging import logging
from ...components.callback_handlers.base_callbacks import RateTriggeredMultiCallbacks
from ...components.callback_handlers.pb_gui_callbacks import (
    PbGUIButtonCallback,
    PbMultiGUISliderSingleCallback,
    PbGUICallback,
)

# pylint: disable=W0212,W0201


class PybulletRobotVisualizer:
    """Visualizer for a PybulletRobot instance to test joint positioning etc.

    See `examples/utils_demo/demo_bullet_robot_visualizer.py`.

    WARNING
    -------
    DO NOT use other GUI utils for pybullet (other sliders, buttons etc), when
    using this. This class uses the `removeAllUserParameters` parameters exposed
    in pybullet, which means it can remove every debug item created externally.

    TODO: fix this to only remove self-created items.
    """

    def __init__(
        self,
        pb_sim_interface: pybullet_robot.BulletRobot,
        joint_lower_lims: Mapping[str, float],
        joint_upper_lims: Mapping[str, float],
        starting_joint_positions: Mapping[str, float] = None,
        ignore_joints_with_str: List[str] = None,
    ):
        """Visualizer for a PybulletSimulatorInterface instance to test joint positioning etc.

        When using with a PybulletRobot or child class object, use the class method `fromBulletRobot`
        to make things easier.

        See `examples/utils_demo/demo_bullet_robot_visualizer.py`

        WARNING
        -------
        DO NOT use other GUI utils for pybullet (other sliders, buttons etc), when
        using this. This class uses the `removeAllUserParameters` parameters exposed
        in pybullet, which means it can remove every debug item created externally.

        TODO: fix this to only remove self-created items.

        Args:
            pb_sim_interface (PybulletSimulatorInterface): The simulator interface instance.
            joint_lower_lims (Mapping[str, float]): Mapping from joint names to joint lower limits.
            joint_upper_lims (Mapping[str, float]): Mapping from joint names to joint upper limits.
            starting_joint_positions (Mapping[str, float], optional): Mapping from joint names to
                starting positions to be used during the visualisation. Defaults to zeros.
            ignore_joints_with_str (List[str], optional): List of substrings to look for in joint
                names, matching joints will be ignored from the joint sliders GUI (useful for
                joints such as wheels). Defaults to ["_wheel"] if None provided.
        """
        self.sim_robot = pb_sim_interface
        self._init_j = starting_joint_positions if starting_joint_positions is not None else {}
        self._ignore_joints_with_str = (
            ["_wheel"] if ignore_joints_with_str is None else ignore_joints_with_str
        )
        self._default_base_pose = copy.deepcopy(self.sim_robot.get_base_pose())
        self._base_pose_target = self.sim_robot.get_base_pose()

        def names_match(name, list_of_words):
            for word in list_of_words:
                if word in name:
                    return True
            return False

        self._enable_stepping = False
        self._j_names = []
        self._j_lower_lims = []
        self._j_upper_lims = []
        for jname in joint_lower_lims.keys():
            if names_match(jname, self._ignore_joints_with_str):
                continue
            self._j_names.append(jname)
            self._j_lower_lims.append(joint_lower_lims[jname])
            self._j_upper_lims.append(joint_upper_lims[jname])
        self._ok_to_run = True
        self._read_button_thread = None

    def run(
        self,
        sim_step_rate: float = 240,
        slider_update_rate: float = 10,
        additional_callbacks: List[Callable[[], None]] = None,
    ):
        """Run the visualizer thread (non-blocking).

        Args:
            sim_step_rate (float, optional): The desired rate (Hz) for calling step simulation
                (This is not the timestep used in the physics of the simulation). Defaults to 240
                (technically this should make it realtime).
            slider_update_rate (float, optional): Update rate (Hz) for reading the joint sliders in
                pybullet GUI to update robot states. Defaults to 10.
        """

        gui_cbs: Dict[str, PbGUICallback] = {}

        if additional_callbacks is None:
            additional_callbacks = []

        for cb in additional_callbacks:
            assert callable(cb)

        if self.sim_robot._in_torque_mode:
            logging.warning(
                "PybulletRobot should be in position control mode to use this visualiser."
                " Changing to position control mode now."
            )
            self.sim_robot.set_position_control_mode()

        self._prev_jvals: np.ndarray = None

        # callbacks for buttons and sliders in the pybullet gui
        def _set_j_cb(j_vals: List[float]):
            if self._prev_jvals is not None and np.allclose(j_vals, self._prev_jvals):
                pass
            else:
                self.sim_robot.reset_actuated_joint_positions(j_vals, self._j_names)
                # setting position control command as well (needed if simulation is switched on)
                self.sim_robot.set_actuated_joint_commands(
                    q=j_vals, actuated_joint_names=self._j_names
                )
                self._prev_jvals = copy.deepcopy(j_vals)

        def _set_base_target_cb(pose_target: List[float]):
            # live update of base pose not being done to allow sim enabling
            self._base_pose_target = (
                pose_target[:3],
                rpy2quat(np.deg2rad(pose_target[3:])),
            )

        def _set_base_pose_cb():
            self.sim_robot.reset_base_pose(self._base_pose_target[0], self._base_pose_target[1])

        def _reset_base_cb():
            self.sim_robot.reset_base_pose(self._default_base_pose[0], self._default_base_pose[1])

        def _place_on_ground_cb():
            jvals = self.sim_robot.get_actuated_joint_positions()
            move_resolution = 0.01
            if not np.any(self.sim_robot.get_contact_states_of_links(self.sim_robot.link_ids)):
                bpos = self.sim_robot.get_base_pose()[0]
                in_collision = False
                while not in_collision:
                    bpos[2] -= move_resolution
                    self.sim_robot.reset_base_pose(
                        position=bpos,
                        orientation=self.sim_robot.get_base_pose()[1],
                    )
                    self.sim_robot.step()
                    in_collision = np.any(
                        self.sim_robot.get_contact_states_of_links(self.sim_robot.link_ids)
                    )

            self.sim_robot._place_robot_on_ground(
                move_resolution=move_resolution,
                default_position=self.sim_robot.get_base_pose()[0],
                default_orientation=self.sim_robot.get_base_pose()[1],
            )
            self.sim_robot.reset_actuated_joint_positions(jvals)

        def _toggle_sim():
            self._enable_stepping = not self._enable_stepping
            if self._enable_stepping:
                # otherwise joints positions will drift
                self.sim_robot.set_actuated_joint_commands(
                    q=self.sim_robot.get_actuated_joint_positions(
                        self.sim_robot.actuated_joint_names
                    ),
                    actuated_joint_names=self.sim_robot.actuated_joint_names,
                )

        def _reset_debug_params(j_mode: str = "default"):
            """Fully reset all GUI stuff (sliders, buttons)"""

            pb.removeAllUserParameters(self.sim_robot.cid)
            ## Below line does not work. pb.removeUserDebugItem does not seem to
            ## work for buttons and sliders!
            # for gui_name in gui_cbs.keys():
            #     gui_cbs[gui_name].remove_gui_object()

            gui_cbs["reset_joints_button"] = PbGUIButtonCallback(
                button_name="reset joints",
                cid=self.sim_robot.cid,
                callback=partial(_reset_debug_params, j_mode="default"),
            )
            gui_cbs["random_joints_button"] = PbGUIButtonCallback(
                button_name="randomise joints",
                cid=self.sim_robot.cid,
                callback=partial(_reset_debug_params, j_mode="random"),
            )
            gui_cbs["zero_joints_button"] = PbGUIButtonCallback(
                button_name="zero all joints",
                cid=self.sim_robot.cid,
                callback=partial(_reset_debug_params, j_mode="zero"),
            )
            j_default_vals = []
            for n, j_name in enumerate(self._j_names):
                if j_mode == "zero":
                    j_default_vals.append(
                        0
                        if self._j_lower_lims[n] <= 0.0 <= self._j_upper_lims[n]
                        else self._j_lower_lims[n]
                    )
                elif j_mode == "random":
                    j_default_vals.append(
                        np.random.uniform(self._j_lower_lims[n], self._j_upper_lims[n])
                    )
                else:
                    j_default_vals.append(0 if j_name not in self._init_j else self._init_j[j_name])
            gui_cbs["triggered_joint_sliders"] = RateTriggeredMultiCallbacks(
                gui_callbacks=[
                    PbMultiGUISliderSingleCallback(
                        slider_names=self._j_names,
                        slider_lower_lims=self._j_lower_lims,
                        slider_upper_lims=self._j_upper_lims,
                        slider_default_vals=j_default_vals,
                        callback=_set_j_cb,
                        cid=self.sim_robot.cid,
                    )
                ],
                rate=slider_update_rate,
            )
            gui_cbs["place_on_ground_button"] = PbGUIButtonCallback(
                button_name="place robot on ground",
                cid=self.sim_robot.cid,
                callback=_place_on_ground_cb,
            )
            gui_cbs["run_sim_button"] = PbGUIButtonCallback(
                button_name="enable/disable simulation",
                cid=self.sim_robot.cid,
                callback=_toggle_sim,
            )

            starting_pos, starting_ori = self.sim_robot.get_base_pose()
            starting_rpy = np.rad2deg(quat2rpy(starting_ori))
            gui_cbs["set_base_sliders"] = PbMultiGUISliderSingleCallback(
                slider_names=[
                    "base_x_pos",
                    "base_y_pos",
                    "base_z_pos",
                    "base_roll",
                    "base_pitch",
                    "base_yaw",
                ],
                slider_lower_lims=[-1, -1, -1, -180, -180, -180],
                slider_upper_lims=[1, 1, 1, 180, 180, 180],
                slider_default_vals=starting_pos.tolist() + starting_rpy.tolist(),
                callback=_set_base_target_cb,
                cid=self.sim_robot.cid,
            )
            gui_cbs["set_base_button"] = PbGUIButtonCallback(
                button_name="set base pose",
                cid=self.sim_robot.cid,
                callback=_set_base_pose_cb,
            )
            gui_cbs["reset_base_button"] = PbGUIButtonCallback(
                button_name="reset base",
                cid=self.sim_robot.cid,
                callback=_reset_base_cb,
            )

        def _read_params_loop():
            """The main gui update loop thread"""

            _reset_debug_params()
            rate_limiter = RateLimiter(sim_step_rate, warn=False)
            logging.info(f"Starting {self.__class__.__name__} instance...")
            while self._ok_to_run:
                for cb in gui_cbs.values():
                    cb.run_once()

                if self._enable_stepping:
                    self.sim_robot.step()

                for cb in additional_callbacks:
                    cb()

                rate_limiter.sleep()

        self._read_button_thread = threading.Thread(target=_read_params_loop)
        self._read_button_thread.start()
        logging.info(f"{self.__class__.__name__} running...")

    def close(self):
        """Close the visualizer."""
        self._ok_to_run = False
        logging.info(f"Attempting to shut down {self.__class__.__name__} instance.")
        if self._read_button_thread is not None:
            self._read_button_thread.join()
        logging.info(f"Closed {self.__class__.__name__} instance.")

    @classmethod
    def fromBulletRobot(
        cls: "PybulletRobotVisualizer",
        pb_robot: PybulletRobot,
        starting_joint_positions: Mapping[str, float] = None,
        ignore_joints_with_str: List[str] = None,
    ) -> "PybulletRobotVisualizer":
        """Visualizer for a PybulletRobot instance to test joint positioning etc.

        See `examples/utils_demo/demo_bullet_robot_visualizer.py`

        Args:
            pb_robot (PybulletRobot): The PybulletRobot instance to load.
            starting_joint_positions (Mapping[str, float], optional): Mapping from joint names to
                starting positions to be used during the visualisation. Defaults to zeros.
            ignore_joints_with_str (List[str], optional): List of substrings to look for in joint
                names, matching joints will be ignored from the joint sliders GUI (useful for
                joints such as wheels). Defaults to ["_wheel"] if None provided.
        """
        pin = pb_robot.get_pinocchio_interface()
        return cls(
            pb_sim_interface=pb_robot._sim_robot,
            joint_lower_lims=dict(zip(pin.actuated_joint_names, pin.actuated_joint_lower_limits)),
            joint_upper_lims=dict(zip(pin.actuated_joint_names, pin.actuated_joint_upper_limits)),
            starting_joint_positions=(
                starting_joint_positions
                if (starting_joint_positions is not None and starting_joint_positions != {})
                else dict(
                    zip(
                        pb_robot._sim_robot.actuated_joint_names,
                        pb_robot._sim_robot.get_actuated_joint_positions(),
                    )
                )
            ),
            ignore_joints_with_str=ignore_joints_with_str,
        )
