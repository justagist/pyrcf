from typing import Mapping, Tuple, List
import copy
import numpy as np
from pybullet_robot import PybulletIKInterface

from .local_planner_base import LocalPlannerBase
from ...utils.filters import second_order_filter
from ...core.types import (
    GlobalMotionPlan,
    RobotState,
    LocalMotionPlan,
    ControlMode,
    PlannerMode,
    Vector3D,
    QuatType,
)


class PybulletIKReferenceInterpolator(LocalPlannerBase):
    """Solves IK using PybulletIKInterface for given end-effector reference and interpolates joint references."""

    def __init__(
        self,
        urdf_path: str,
        floating_base: bool = False,
        starting_base_position: Vector3D = np.zeros(3),
        starting_base_orientation: QuatType = np.array([0, 0, 0, 1]),
        starting_joint_positions: List[float] = None,
        joint_names_order: List[str] = None,
        filter_gain=0.05,
        blind_mode: bool = True,
        ee_names: List[str] = None,
        max_constraint_force: float = 10000,
        solver_erp: float = None,
        **pb_ik_kwargs,
    ):
        """A "local planner" that simply solves IK for given end-effector reference
        and interpolates joint references to achieve the target EE pose. Uses
        PybulletIKInterface from pybullet_robot package to solve IK.

        Args:
            urdf_path (str): Path to urdf of the robot.
            floating_base (bool, optional): Whether the robot has fixed base or floating base.
                Defaults to False.
            starting_base_position (Vector3D, optional): Starting base position of the robot.
                Defaults to np.zeros(3).
            starting_base_orientation (QuatType, optional): Starting base orientation of the robot.
                Defaults to np.array([0, 0, 0, 1]).
            starting_joint_positions (List[float], optional): Starting joint positions of the
                robot. Defaults to None.
            joint_names_order (List[str], optional): Order of joint names to be used. This will be
                the order of the output of the IK as well. Defaults to None (use default order from
                BulletRobot instance when loading this urdf).
            filter_gain (float, optional): Smoothing gain for second-order filter
                interpolator. Defaults to 0.05.
            blind_mode (bool, optional): If set to False, will not use joint state
                feedback for generating reference (will continue from previous target).
                Defaults to True.
            ee_names (List[str], optional): If list of end-effector names are provided, only
                IK tasks for these will be added. Defaults to None (all EE's whose references
                are provided in the global plan message will be tracked).
            max_constraint_force (float, optional): Max force this constraint is allowed to apply
                to pull the link. Defaults to 10000 (setting `None` will use pybullet default).
            solver_erp (float, optional): Error reduction parameter for this constraint. Defaults
                to None (use pybullet default).
            **pb_ik_kwargs (optional): Additional keywork arguments to be used while creating the
                PybulletIKInterface object.
        """
        self._pb_ik = PybulletIKInterface(
            urdf_path=urdf_path,
            floating_base=floating_base,
            starting_base_position=starting_base_position,
            starting_base_orientation=starting_base_orientation,
            starting_joint_positions=starting_joint_positions,
            joint_names_order=joint_names_order,
            **pb_ik_kwargs,
        )
        self._alpha = filter_gain
        self._blind_mode = blind_mode
        self._max_force = max_constraint_force
        self._erp = solver_erp
        self._ee_names = ee_names
        self._output_plan: LocalMotionPlan = None
        self._prev_ee_pose_targets: Mapping[str, Tuple[Vector3D, QuatType]] = {}
        self._prev_ref: np.ndarray = None

    def generate_local_plan(
        self,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        t: float,
        dt: float,
    ) -> LocalMotionPlan:

        if global_plan.planner_mode != PlannerMode.CUSTOM:
            self._output_plan = None
            return LocalMotionPlan()

        if global_plan.end_effector_references.ee_names is not None:
            if self._output_plan is None:
                self._output_plan = LocalMotionPlan(control_mode=ControlMode.CONTROL)
                self._output_plan.joint_references.joint_names = copy.deepcopy(
                    self._pb_ik.joint_name_order
                )
                if self._blind_mode:
                    self._prev_ref = np.array(
                        [
                            robot_state.joint_states.get_state_of(jname)[0]
                            for jname in self._pb_ik.joint_name_order
                        ]
                    )
            for ee_name in global_plan.end_effector_references.ee_names:

                if self._ee_names is not None and ee_name not in self._ee_names:
                    # if only specific end-effectors are to be tracked, ignore the end-effectors
                    # not in the list even if they have a global plan target pose
                    continue

                target_pose, _, _, _ = global_plan.end_effector_references.get_state_of(
                    ee_name=ee_name
                )
                if target_pose is None:
                    continue
                if not self._pb_ik.frame_task_exists(frame_name=ee_name):
                    self._pb_ik.add_frame_task(frame_name=ee_name)
                if ee_name in self._prev_ee_pose_targets:
                    p, q = self._prev_ee_pose_targets[ee_name]
                    if np.allclose(p, target_pose.position) and np.allclose(
                        q, target_pose.orientation
                    ):
                        continue
                self._prev_ee_pose_targets[ee_name] = [
                    target_pose.position.copy(),
                    target_pose.orientation.copy(),
                ]
                self._pb_ik.update_frame_task(
                    frame_name=ee_name,
                    target_position=target_pose.position.copy(),
                    target_orientation=target_pose.orientation.copy(),
                    erp=self._erp,
                    max_force=self._max_force,
                )
            sol = self._pb_ik.get_ik_solution()
            q = sol.q if not self._pb_ik.floating_base else sol.q[7:]
            if self._blind_mode:
                self._output_plan.joint_references.joint_positions = second_order_filter(
                    current_value=self._prev_ref,
                    desired_value=q,
                    gain=self._alpha,
                )
                self._output_plan.joint_references.joint_velocities = (
                    self._output_plan.joint_references.joint_positions - self._prev_ref
                ) / (dt + 1e-5)
                self._prev_ref = self._output_plan.joint_references.joint_positions.copy()
            else:
                curr_joints = np.array(
                    [
                        robot_state.joint_states.get_state_of(jname)[0]
                        for jname in global_plan.joint_references.joint_names
                    ]
                )
                self._output_plan.joint_references.joint_positions = second_order_filter(
                    current_value=curr_joints,
                    desired_value=q,
                    gain=self._alpha,
                )
                self._output_plan.joint_references.joint_velocities = (
                    self._output_plan.joint_references.joint_positions - curr_joints
                ) / dt

        if self._output_plan is None:
            return LocalMotionPlan()

        return self._output_plan
