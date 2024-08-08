from typing import List, Tuple, Callable, TypeAlias
import pybullet as pb
import numpy as np

from ...components.callback_handlers.base_callbacks import CustomCallbackBase

QuatType: TypeAlias = np.ndarray
"""Numpy array representating quaternion in format [x,y,z,w]"""
Vector3D: TypeAlias = np.ndarray
"""Numpy array representating 3D cartesian vector in format [x,y,z]"""


class PybulletDebugRobot:
    """A debugger robot visualiser for bullet robots. These robots are meant to be used for
    visualising ideal motion plans etc. for debugging (e.g. as a control loop debugger,
    implemented in `utils.ctrl_loop_debuggers.BulletRobotPlanDebugger`)."""

    def __init__(
        self,
        urdf_path: str,
        cid: int = 0,
        base_position: np.ndarray = np.zeros(3),
        base_orientation: np.array = np.array([0, 0, 0, 1]),
        rgba: Tuple[float, float, float, float] | None = (0, 0, 0, 0.3),
    ):
        """A debugger robot visualiser for bullet robots. These robots are meant to be used for
        visualising ideal motion plans etc. for debugging (e.g. as a control loop debugger,
        implemented in `utils.ctrl_loop_debuggers.BulletRobotPlanDebugger`).

        Args:
            urdf_path (str): Path to robot urdf
            cid (int, optional): The client interface to connect to an existing pybullet server.
                Defaults to 0.
            base_position (np.ndarray, optional): Starting base position for the robot. Defaults to
                np.zeros(3).
            base_orientation (np.array, optional): Starting base orientation quaternion for the
                robot. Defaults to np.array([0, 0, 0, 1]).
            rgba (Tuple[float, float, float, float], optional): The RGBA tuple for the debugger
                robot visualiser. Defaults to (0, 0, 0, 0.3).
        """
        self._cid = cid
        self._viz_robot_id = pb.loadURDF(
            urdf_path,
            basePosition=base_position,
            baseOrientation=base_orientation,
            physicsClientId=self._cid,
            useFixedBase=True,
            flags=pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )

        self._link_ids = [-1] + list(
            range(pb.getNumJoints(self._viz_robot_id, physicsClientId=cid))
        )
        self._visible = True
        for link_id in self._link_ids:
            pb.setCollisionFilterGroupMask(
                self._viz_robot_id, link_id, 0, 0, physicsClientId=self._cid
            )
            pb.changeDynamics(
                self._viz_robot_id,
                link_id,
                mass=0,  # non-dynamic objects have mass 0
                lateralFriction=0,
                spinningFriction=0,
                rollingFriction=0,
                restitution=0,
                linearDamping=0,
                angularDamping=0,
                contactStiffness=0,
                contactDamping=0,
                jointDamping=0,
                maxJointVelocity=0,
                physicsClientId=self._cid,
            )

        if rgba is not None:
            self.set_visual_rgba(rgba=rgba)
            self._latest_rgba = rgba
        else:
            self._latest_rgba = [0, 0, 0, 0.3]

        _pb_keys = [
            "jointIndex",
            "jointName",
            "jointType",
        ]

        num_js = pb.getNumJoints(self._viz_robot_id, physicsClientId=cid)
        j_infos = [[] for _ in range(len(_pb_keys))]
        for i in range(num_js):
            j_info_tuple = pb.getJointInfo(self._viz_robot_id, i, physicsClientId=cid)
            for jinfo, info_item in zip(j_infos, j_info_tuple):
                if isinstance(info_item, bytes):
                    info_item = info_item.decode()
                jinfo.append(info_item)
        jinfo = dict(zip(_pb_keys, j_infos))
        self._actuated_joint_names = []

        for n, name in enumerate(jinfo["jointName"]):
            jtype = jinfo["jointType"][n]
            if jtype == pb.JOINT_FIXED:
                continue
            self._actuated_joint_names.append(name)

        joint_ids = jinfo["jointIndex"]
        self._joint_name_to_index = dict(zip(jinfo["jointName"], joint_ids))

    def disable_visualisation(self):
        self.set_visual_rgba(rgba=[0, 0, 0, 0])

    def set_visual_rgba(self, rgba: Tuple[float, float, float, float]):
        for link_id in self._link_ids:
            pb.changeVisualShape(
                self._viz_robot_id,
                link_id,
                rgbaColor=rgba,
                physicsClientId=self._cid,
            )
        self._visible = rgba[3] > 0
        if self._visible:
            self._latest_rgba = rgba

    def enable_visualisation(self):
        self.set_visual_rgba(rgba=self._latest_rgba)

    def toggle_visualisation(self):
        if self._visible:
            self.disable_visualisation()
        else:
            self.enable_visualisation()

    def set_base_pose(self, position: np.ndarray, orientation: np.ndarray):
        pb.resetBasePositionAndOrientation(
            self._viz_robot_id, position, orientation, physicsClientId=self._cid
        )

    def set_joint_positions(self, joint_positions: np.ndarray, joint_names: List[str] = None):
        if joint_names is None:
            joint_names = self._actuated_joint_names
        for name, pos in zip(joint_names, joint_positions):
            pb.resetJointState(
                self._viz_robot_id,
                self._joint_name_to_index[name],
                pos,
                physicsClientId=self._cid,
            )

    def close(self):
        try:
            pb.removeBody(
                self._viz_robot_id,
                physicsClientId=self._cid,
            )
        except pb.error:
            pass


class PbDebugRobotWithJointCallback(CustomCallbackBase):
    def __init__(
        self,
        urdf_path: str,
        cid: int,
        joint_names: List[str],
        get_joint_positions_callback: Callable[[], np.ndarray],
        get_base_pose_callback: Callable[[], Tuple[Vector3D, QuatType]] = lambda: None,
        base_position: np.ndarray = np.zeros(3),
        base_orientation: np.array = np.array([0, 0, 0, 1]),
        rgba: Tuple[float, float, float, float] | None = (0, 0, 0, 0.3),
    ):
        self._viz_robot = PybulletDebugRobot(
            urdf_path=urdf_path,
            cid=cid,
            base_position=base_position,
            base_orientation=base_orientation,
            rgba=rgba,
        )
        self._joint_names = joint_names
        self._js_callback = get_joint_positions_callback
        self._bpose_callback = get_base_pose_callback
        assert callable(self._js_callback)
        assert callable(self._bpose_callback)

    def run_once(self):
        """This method has to be called for setting the joint positions. This will
        run the callback method to get the desired joint positions."""
        jvals = self._js_callback()
        if jvals is not None:
            self._viz_robot.set_joint_positions(
                joint_positions=jvals, joint_names=self._joint_names
            )
        bpose = self._bpose_callback()
        if bpose is not None:
            self._viz_robot.set_base_pose(position=bpose[0], orientation=bpose[1])

    def cleanup(self):
        del self._viz_robot
