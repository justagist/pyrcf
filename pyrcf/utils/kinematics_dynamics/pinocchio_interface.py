"""Provides utilities for kinematics and dynamics parameters retrievals.

This is a standalone file.
pypi dependencies: [numpy, pin]

FEATURES:
    - Retrieve all dynamics and kinematics info for any robot (urdf).
    - All methods become available after a single call to the `update()` method of the class, which
        will compute all required kinematics and dynamics data, which can be retreived using the
        other methods in the class.
    - Handles continuous joints in addition to revolute and prismatic joints.
    - Extends the default pinocchio.RobotWrapper class, so all methods from base class still
        available (if required).

TODO:
    - [x] better way to identify continuous joints is to use pin.model.joints[id].nq and
        pin.model.joints[id].nv values.
        These should be 2 & 1 respectively for continuous joints, 6 & 5 for free-floating, and
        1 & 1 for others.
        NOTE: 'composite joints' can have different nq and nv numbers.
        - [x] also can use joints.shortname() to get joint type directly from this list:
        https://github.com/stack-of-tasks/pinocchio/blob/0caf0ca4d07e63834cdc420c703993662c59e01b/include/pinocchio/multibody/joint/joint-collection.hpp#L24-L67
    - [ ] handle additional joint types.
"""

from typing import List, Tuple, TypeAlias, Sequence, Mapping
import pinocchio
import numpy as np


QuatType: TypeAlias = np.ndarray
"""Numpy array representating quaternion in format [x,y,z,w]"""
Vector3D: TypeAlias = np.ndarray
"""Numpy array representating 3D cartesian vector in format [x,y,z]"""


class PinocchioInterface(pinocchio.RobotWrapper):
    """A pinocchio.RobotWrapper extension to get robot kinematics and dynamics values easily.

    This class cannot deal with passive joints, spherical joints or other composite joints.
    It works fine with regular revolute and prisimatic joints, as well as handles continuous
    joints.
    """

    @staticmethod
    def configuration_handle_continuous_joints(
        q: np.ndarray, continuous_joint_q_ids: Sequence[float]
    ) -> np.ndarray:
        """Static method to update the provided pinocchio q vector to handle continuous joints.

        This is to be used when the original indices are filled with the joint
        position values. These will then be replaced with cos(theta) and sin(theta)
        values in sequence at each of these indices (as required by pinocchio)
        for continuous joints only.

        Args:
            q (np.ndarray): The original values where the values at indices `continuous_joint_q_ids`
                is filled with the actual value of joint position encoder.
            continuous_joint_q_ids (Sequence[float]): The indices in the q vector which are
                continuous joints.

        Returns:
            np.ndarray: Updated q vector with appropriate 2D representation for continuous joints.
        """
        if len(continuous_joint_q_ids) == 0:
            return q

        in_q = q.copy()
        sin_q_ids = np.array(continuous_joint_q_ids) + 1
        continuous_j_vals = q[continuous_joint_q_ids].copy()
        in_q[continuous_joint_q_ids] = np.cos(continuous_j_vals)
        in_q[sin_q_ids] = np.sin(continuous_j_vals)
        return in_q

    @staticmethod
    def recover_generalised_pos_from_pinocchio_configuration(
        q: np.ndarray,
        continuous_joint_q_ids: Sequence[float] = None,
    ) -> np.ndarray:
        """Static method to get the generalised coordinate position vector from pinocchio q vector.

        This function is mainly for handling situations where the robot has continuous joints.
        Pinocchio uses a 2D representation for a single continuous joint, and this function
        recovers single joint position values for such joints.

        Args:
            q (np.ndarray): Configuration vector in the format used by Pinocchio.
            continuous_joint_q_ids (Sequence[float], optional): The joint ids of continuous joints.
                Defaults to [].

        Returns:
            np.ndarray: Output generalised coordinate vector after removing pinocchio's
                continuous joint representation.
        """
        in_q = q.copy()
        if continuous_joint_q_ids is not None and len(continuous_joint_q_ids) > 0:
            sin_ids = np.array(continuous_joint_q_ids) + 1
            in_q[continuous_joint_q_ids] = np.arctan2(q[sin_ids], q[continuous_joint_q_ids])
        return in_q

    def __init__(
        self,
        urdf_filename: str,
        floating_base: bool = True,
        verbose: bool = True,
        ee_names: List[str] = None,
    ):
        """Constructor.
        A pinocchio.RobotWrapper extension to get robot kinematics and dynamics values easily.

        This class cannot deal with passive joints, spherical joints or other composite joints.
        It works fine with regular revolute and prisimatic joints, as well as handles continuous
        joints.

        Args:
            urdf_filename (str): path to urdf file of the robot.
            floating_base (bool, optional): Whether the robot has fixed base or floating base.
                Defaults to True.
            verbose (bool, optional): Verbosity flag. Defaults to True.
            ee_names (List[str], optional): List of end-effector links for the robot.
                Defaults to empty list.
        """
        self.floating_base = floating_base
        if self.floating_base is True:
            self.model = pinocchio.buildModelFromUrdf(
                urdf_filename, pinocchio.JointModelFreeFlyer()
            )
        else:
            self.model = pinocchio.buildModelFromUrdf(urdf_filename)
        super().__init__(self.model, verbose=verbose)

        self.name = self.model.name
        """Name of the robot model."""

        self.total_mass = pinocchio.computeTotalMass(self.model)
        self.num_joints = self.model.njoints
        self.joint_names = self.model.names.tolist()
        self.joint_lower_limits: List[float] = np.array(self.model.lowerPositionLimit)
        self.joint_upper_limits: List[float] = np.array(self.model.upperPositionLimit)
        self.joint_neutral_position = (self.joint_lower_limits + self.joint_upper_limits) / 2
        self.joint_velocity_limit: List[float] = np.array(self.model.velocityLimit)

        self.actuated_joint_names: List[str] = []
        self.actuated_joint_q_ids: List[int] = []
        """index of joints in the generalised coordinate vector (q)"""

        self.q = np.zeros(self.model.nq)  # will be filled with first call to update()
        """The pinocchio q vector (generalised coordinates)."""

        if self.floating_base:
            # set floating joint orientation to unit quaternion
            self.q[7] = 1.0

        self.v = np.zeros(self.model.nv)  # will be filled with first call to update()
        """The pinocchio v vector (generalised velocity)."""
        self.a = np.zeros(self.model.nv)  # will be filled with first call to update()
        """The pinocchio a output vector (generalised acceleration). NOTE: never filled!"""

        self.actuated_joint_v_ids: List[int] = []
        """index of joints in the generalised velocity vector (v)"""

        self.continuous_joint_q_ids: List[int] = []
        """Indices corresponding to continuous joints in the pinocchio configuration vector q.
        In pinocchio, continuous joints are represented using 2 values [cos(theta), sin(theta)].
        This list contains the index to the first (cos) value (add 1 to the index to get the next).
        """

        self.continuous_joint_names: List[str] = []

        def joint_is_continuous(joint: pinocchio.pinocchio_pywrap.JointModel):
            """Check if the pinocchio joint is continuous (as defined in pinocchio)."""
            if (
                joint.nq == 2
                and joint.nv == 1
                # this below check alone should be enough; using nq, nv just to be sure
                and joint.shortname()
                in [
                    "JointModelRUBX",
                    "JointModelRUBY",
                    "JointModelRUBZ",
                    "JointModelRevoluteUnboundedUnaligned",  # unaligned continuous joints
                ]
            ):
                return True
            return False

        for n, name in enumerate(self.joint_names):
            if name not in ["universe", "root_joint"]:
                curr_j_id = self.model.idx_qs[self.model.getJointId(name)]
                curr_v_id = self.model.idx_vs[self.model.getJointId(name)]
                self.actuated_joint_names.append(name)
                self.actuated_joint_q_ids.append(curr_j_id)
                self.actuated_joint_v_ids.append(curr_v_id)
                # To handle continuous joint in the generalised coordinates (q), save these ids
                # separately pinocchio represents continuous joints with 2 numbers (cos(theta),
                # sin(theta))
                if joint_is_continuous(self.model.joints[n]):
                    self.continuous_joint_q_ids.append(curr_j_id)
                    self.continuous_joint_names.append(name)
                # NOTE (@ssidhik): handle composite joints, passive joints, etc.

        self.num_of_actuated_joints = len(self.actuated_joint_names)

        self.actuated_joint_q_ids = np.array(self.actuated_joint_q_ids)
        self.actuated_joint_v_ids = np.array(self.actuated_joint_v_ids)

        additional_dims_ = 7 if self.floating_base else 0
        expected_nq_ = (
            self.num_of_actuated_joints + len(self.continuous_joint_q_ids) + additional_dims_
        )

        assert (
            expected_nq_ == self.nq
        ), f"Could not match expected nq for model. Required model nq: {self.nq}; Current nq obtained: {expected_nq_}."

        self.actuated_joint_name_to_q_index: Mapping[str, int] = dict(
            zip(self.actuated_joint_names, self.actuated_joint_q_ids)
        )
        """Mapping from joint name to index of the joint configuration value in the pinocchio q
        vector."""

        self.actuated_joint_name_to_v_index: Mapping[str, int] = dict(
            zip(self.actuated_joint_names, self.actuated_joint_v_ids)
        )
        """Mapping from joint name to index of the joint velocity value in the pinocchio v
        vector."""

        self.actuated_joint_lower_limits = self.get_joint_positions_from_configuration(
            self.joint_lower_limits
        )
        self.actuated_joint_upper_limits = self.get_joint_positions_from_configuration(
            self.joint_upper_limits
        )
        self.actuated_joint_neutral_position = (
            self.actuated_joint_lower_limits + self.actuated_joint_upper_limits
        ) / 2

        self.actuated_joint_limits = {}
        """mapping from {'joint_name' -> [lower_lim, upper_lim]}"""
        for n, name in enumerate(self.actuated_joint_names):
            self.actuated_joint_limits[name] = [
                self.actuated_joint_lower_limits[n],
                self.actuated_joint_upper_limits[n],
            ]

        self.num_of_frames = self.model.nframes
        self.frame_names = [frame.name for frame in self.model.frames]
        self.frame_ids = [self.model.getFrameId(frame_name) for frame_name in self.frame_names]

        self.ee_names = ee_names if ee_names is not None else []
        """Names of frames that are used as end-effectors."""

        if verbose:
            self.print_model_info()

    def print_model_info(self):
        print("\n")
        print("*" * 100 + "\nPinocchio Model Info " + "\u2193 " * 20 + "\n" + "*" * 100)
        print("name: ", self.model.name)
        print("total_mass:", self.total_mass)
        print("floating base:", self.floating_base)
        print("nq: ", self.nq)
        print("nv: ", self.nv)
        print("joint names:", self.num_joints, self.joint_names)
        print("End-effectors:", self.ee_names)
        print(
            f"Number of moveable joints: {len(self.actuated_joint_q_ids)}, of which "
            f"{len(self.continuous_joint_q_ids)} are continuous joints (with 2 values in pinochio "
            "generalised coordinates)."
        )
        print(
            "actuated joint names: ",
            self.actuated_joint_names,
        )
        print(
            "continuous joint names: ",
            self.continuous_joint_names,
        )
        print("frame names:", self.num_of_frames, self.frame_names)
        print("*" * 100 + "\nPinocchio Model Info " + "\u2191 " * 20 + "\n" + "*" * 100)
        print("\n")

    def update(
        self,
        global_base_position: Vector3D,
        global_base_quaternion: QuatType,
        local_base_velocity_linear: Vector3D,
        local_base_velocity_angular: Vector3D,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_order: List[str] = None,
    ):
        """Do all forward kinematics and dynamics computations for the given robot states.

        Also updates the internal configuration vectors (self.q, self.v).

        Args:
            global_base_position (Vector3D): Position of robot base in the world.
            global_base_quaternion (QuatType): Orientation quaternion (x,y,z,w) of the
                robot base in the world.
            local_base_velocity_linear (Vector3D): Linear velocity of the robot base in the local
                base frame.
            local_base_velocity_angular (Vector3D): Angular velocity of the robot base in the local
                base frame.
            joint_positions (np.ndarray): Current joint positions of the robot.
            joint_velocities (np.ndarray): Current joint velocities of the robot.
            joint_order (List[str], optional): Joint names in the order the position values were
                given. If None provided, uses default order found by pinocchio (urdf order).
        """
        self.q, self.v = self.get_pinocchio_q_and_v_from_robot_state(
            global_base_position=global_base_position,
            global_base_quaternion=global_base_quaternion,
            local_base_velocity_linear=local_base_velocity_linear,
            local_base_velocity_angular=local_base_velocity_angular,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_order=joint_order,
        )
        self.a = np.zeros(self.model.nv)

        pinocchio.forwardKinematics(self.model, self.data, self.q, self.v, self.a)
        pinocchio.framesForwardKinematics(self.model, self.data, self.q)
        pinocchio.computeJointJacobians(self.model, self.data, self.q)
        pinocchio.computeCentroidalMap(self.model, self.data, self.q)
        pinocchio.computeCentroidalMapTimeVariation(self.model, self.data, self.q, self.v)
        pinocchio.ccrba(self.model, self.data, self.q, self.v)

    def get_frame_id(self, frame_name: str) -> int:
        return self.model.getFrameId(frame_name)

    def get_com_position(self) -> Vector3D:
        return self.com(self.q)

    def get_com_velocity(self) -> Vector3D:
        return self.vcom(self.q, self.v)

    def get_centroidal_mass(self) -> float:
        return self.data.Ig.mass

    def get_gravity_vector(self) -> np.ndarray:
        return self.gravity(q=self.q)

    def get_centroidal_inertia(self) -> np.ndarray:
        return self.data.Ig.inertia

    def get_centroidal_momentum_map(self) -> np.ndarray:
        return self.data.Ag

    def get_centroidal_momentum_map_linear(self) -> np.ndarray:
        return self.get_centroidal_momentum_map()[:3, :]

    def get_centroidal_momentum_map_angular(self) -> np.ndarray:
        return self.get_centroidal_momentum_map()[-3:, :]

    def get_inertia_matrix(self) -> np.ndarray:
        return self.mass(self.q)

    def get_nonlinear_effects(self) -> np.ndarray:
        return self.nle(self.q, self.v)

    def get_joint_positions(self, joint_names: List[str] = None) -> np.ndarray:
        """Get the current joint positions using the latest joint configuration vector.

        This method also handles continuous joints and extracts joint position values for
        continuous joints from the 2D configuration representation used in pinocchio.

        Args:
            joint_names (List[str], optional): Optional list of joints/joint order. Defaults to None
                (uses default joint order).

        Returns:
            np.ndarray: Values of joint position/orientation for specified/all joints.
        """
        return self.get_joint_positions_from_configuration(q=self.q, joint_names=joint_names)

    def get_joint_positions_from_configuration(
        self, q: np.ndarray, joint_names: List[str] = None
    ) -> np.ndarray:
        """Get the joint positions from the provided pinocchio configuration vector.

        This method also handles continuous joints and extracts joint position values for
        continuous joints from the 2D configuration representation used in pinocchio.

        NOTE: This assumes the provided q vector has the same order as the self.q values in this
        class.

        Args:
            q (np.ndarray): Pinocchio configuration vector from which joint values are to be
                extracted. Assumes the provided q vector has the same order as the self.q values in
                this class.
            joint_names (List[str], optional): Optional list of joints/joint order. Defaults to None
                (uses default joint order).

        Returns:
            np.ndarray: Values of joint position/orientation for specified/all joints.
        """
        q = self.get_generalised_coordinates_from_q(q)
        if joint_names is not None and joint_names != self.actuated_joint_names:
            indexes = [self.actuated_joint_name_to_q_index[jname] for jname in joint_names]
            return q[indexes]
        return q[self.actuated_joint_q_ids]

    def get_generalised_coordinates_from_q(self, q: np.ndarray) -> np.ndarray:
        """Get the generalised coordinate representation for the state of the robot.

        This is going to be ordered joint position values in case of fixed-base robots,
        and concatenation of base position, base quaternion (x,y,z,w) and joint positions
        for floating base systems.

        Args:
            q (np.ndarray): Input Pinocchio q vector.

        Returns:
            np.ndarray: output generalised coordinate state representation of this robot.
        """
        assert (
            len(q) == self.nq
        ), f"Input to this function should be a vector of size {self.nq}; got {len(q)}"
        return PinocchioInterface.recover_generalised_pos_from_pinocchio_configuration(
            q=q,
            continuous_joint_q_ids=self.continuous_joint_q_ids,
        )

    def get_pinocchio_q_and_v_from_robot_state(
        self,
        global_base_position: Vector3D,
        global_base_quaternion: QuatType,
        local_base_velocity_linear: Vector3D,
        local_base_velocity_angular: Vector3D,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_order: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get pinocchio configuration vector (q) and velocity (v) from robot state data.

        Also handles continuous joints.

        Args:
            global_base_position (Vector3D): Position of robot base in the world.
            global_base_quaternion (QuatType): Orientation quaternion (x,y,z,w) of the
                robot base in the world.
            local_base_velocity_linear (Vector3D): Linear velocity of the robot base in the local
                base frame.
            local_base_velocity_angular (Vector3D): Angular velocity of the robot base in the local
                base frame.
            joint_positions (np.ndarray): Current joint positions of the robot.
            joint_velocities (np.ndarray): Current joint velocities of the robot.
            joint_order (List[str], optional): Joint names in the order the position values were
                given. If None provided, uses default order found by pinocchio (urdf order).

        Returns:
            Tuple[np.ndarray, np.ndarray]: The q and v vectors that can be used for pinocchio
                operations.
        """
        q = np.zeros(self.model.nq)
        v = np.zeros(self.model.nv)

        if joint_order is not None and joint_order != self.actuated_joint_names:
            assert len(joint_order) == len(self.actuated_joint_names)
            for n, name in enumerate(joint_order):
                q[self.actuated_joint_name_to_q_index[name]] = joint_positions[n]
                v[self.actuated_joint_name_to_v_index[name]] = joint_velocities[n]
        else:
            q[self.actuated_joint_q_ids] = joint_positions
            v[self.actuated_joint_v_ids] = joint_velocities

        if self.floating_base:
            q[:3] = global_base_position
            q[3:7] = global_base_quaternion
            v[:3] = local_base_velocity_linear
            v[3:6] = local_base_velocity_angular

        q = PinocchioInterface.configuration_handle_continuous_joints(
            q=q, continuous_joint_q_ids=self.continuous_joint_q_ids
        )

        return q, v

    def get_frame_position(self, frame_name: str, reference_frame: str = "world") -> Vector3D:
        if reference_frame == "world":
            return self.framePlacement(self.q, self.get_frame_id(frame_name)).translation
        else:
            transform = self.get_frame_transformation(frame_name, reference_frame=reference_frame)
            return transform[:3, 3]

    def get_global_frame_pose(self, frame_name: str) -> Tuple[Vector3D, QuatType]:
        frame_transform = self.framePlacement(self.q, self.get_frame_id(frame_name))
        return (
            frame_transform.translation,
            pinocchio.Quaternion(frame_transform.rotation).coeffs(),
        )

    def get_ee_poses(self, ee_names: List[str] = None) -> List[Tuple[Vector3D, QuatType]]:
        if ee_names is None:
            ee_names = self.ee_names
        return [self.get_global_frame_pose(frame_name=ee_name) for ee_name in ee_names]

    def get_ee_velocities(self, ee_names: List[str] = None) -> List[Tuple[Vector3D, Vector3D]]:
        if ee_names is None:
            ee_names = self.ee_names
        return [
            (
                self.get_frame_velocity_linear(frame_name=ee_name),
                self.get_frame_velocity_angular(frame_name=ee_name),
            )
            for ee_name in ee_names
        ]

    def get_frame_rotation(self, frame_name: str, reference_frame: str = "world") -> np.ndarray:
        if reference_frame == "world":
            return self.framePlacement(self.q, self.get_frame_id(frame_name)).rotation
        else:
            transform = self.get_frame_transformation(frame_name, reference_frame=reference_frame)
            return transform[:3, :3]

    def get_frame_quaternion(self, frame_name: str, reference_frame: str = "world") -> QuatType:
        return pinocchio.Quaternion(self.get_frame_rotation(frame_name, reference_frame)).coeffs()

    def get_frame_transformation(
        self, frame_name: str, reference_frame: str = "world"
    ) -> np.ndarray:
        def homogeneous_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
            mat = np.eye(4)
            mat[:3, :3] = rotation
            mat[:3, 3] = translation
            return mat

        pos = self.get_frame_position(frame_name)
        rot = self.get_frame_rotation(frame_name)
        if reference_frame == "world":
            return homogeneous_matrix(rot, pos)

        transform = homogeneous_matrix(rot, pos)
        pos_reference = self.get_frame_position(reference_frame)
        rot_reference = self.get_frame_rotation(reference_frame)
        transform_reference = homogeneous_matrix(rot_reference, pos_reference)
        local_transformation = np.linalg.pinv(transform_reference) @ transform
        return local_transformation

    def get_frame_velocity_linear(
        self, frame_name: str, reference_frame: str = "world"
    ) -> Vector3D:
        if reference_frame == "world":
            return (
                self.get_frame_rotation(frame_name)
                @ self.frameVelocity(self.q, self.v, self.get_frame_id(frame_name)).linear
            )
        else:
            raise RuntimeError("reference_frame not implemented")

    def get_frame_velocity_angular(
        self, frame_name: str, reference_frame: str = "world"
    ) -> Vector3D:
        if reference_frame == "world":
            return (
                self.get_frame_rotation(frame_name)
                @ self.frameVelocity(self.q, self.v, self.get_frame_id(frame_name)).angular
            )
        else:
            raise RuntimeError("reference_frame not implemented")

    def get_frame_acceleration_linear(
        self, frame_name: str, reference_frame: str = "world"
    ) -> Vector3D:
        if reference_frame == "world":
            return (
                self.get_frame_rotation(frame_name)
                @ self.frameAcceleration(
                    self.q, self.v, self.a, self.get_frame_id(frame_name)
                ).linear
            )
        else:
            raise RuntimeError("reference_frame not implemented")

    def get_frame_acceleration_angular(
        self, frame_name: str, reference_frame: str = "world"
    ) -> Vector3D:
        if reference_frame == "world":
            return (
                self.get_frame_rotation(frame_name)
                @ self.frameAcceleration(
                    self.q, self.v, self.a, self.get_frame_id(frame_name)
                ).angular
            )
        else:
            raise RuntimeError("reference_frame not implemented")

    def get_frame_jacobian(self, frame_name: str, reference_frame: str = "world") -> np.ndarray:
        if reference_frame == "world":
            return self.getFrameJacobian(
                self.get_frame_id(frame_name),
                pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
        else:
            raise RuntimeError("reference_frame not implemented")

    def get_frame_jacobian_linear(
        self, frame_name: str, reference_frame: str = "world"
    ) -> np.ndarray:
        if reference_frame == "world":
            return self.getFrameJacobian(
                self.get_frame_id(frame_name),
                pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, :]
        else:
            raise RuntimeError("reference_frame not implemented")

    def get_frame_jacobian_angular(
        self, frame_name: str, reference_frame: str = "world"
    ) -> np.ndarray:
        if reference_frame == "world":
            return self.getFrameJacobian(
                self.get_frame_id(frame_name),
                pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[-3:, :]
        else:
            raise RuntimeError("reference_frame not implemented")
