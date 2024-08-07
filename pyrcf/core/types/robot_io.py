from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple, Mapping, Any
import copy
from numbers import Number

from .tf_types import Pose3D, Twist, Vector3D
from ..logging import logging


@dataclass
class EndEffectorStates:
    """State estimate data relating to end-effectors of the robot."""

    ee_names: List[str] = None
    """Names of the end-effector frames."""
    ee_poses: List[Pose3D] = None
    """Pose of each end-effector.
    TODO: later, this may need to be of type: `List[FrameMotion]`.
    """
    ee_twists: List[Twist] = None
    """Twist for each end-effector"""
    contact_states: List[int] = None
    """Binary values denoting if end-effector(s) are in contact. 1 for contact, 0 for no-contact.
    Order is same as that of ee_names."""
    contact_forces: List[Vector3D] = None
    """A list of 3D vectors representing contact forces per end effector.

    Each row provides the x,y,z forces on one end_effector (order same as ee_names).
    """

    # internal mapping stored to make retrieval by name easier
    _ee_name_to_idx: Mapping[str, int] = field(default=None, init=False, repr=False)

    def get_state_of(
        self, ee_name: str
    ) -> Tuple[Pose3D | None, Twist | None, int | None, Vector3D | None]:
        """Get the end-effector states (pose, contact state, contact force) of the specified
        end-effector frame.

        This method is equivalent to directly using the `__getitem__` of this object. I.e.
        The two lines below are equivalent:
            >> p, v, c, f = ee_states.get_state_of(ee_name)
            >> p, v, c, f = ee_states[ee_name]

        Args:
            ee_name (str): The name of the end-effector frame.

        Raises:
            AttributeError: Raised if `ee_names` attribute was not defined for this object.
            KeyError: If the specified `ee_name` is not in the `ee_names` of this object.

        Returns:
            Tuple[Pose3D | None, Twist | None, int | None, Vector3D | None]: End-effector pose,
                twist, contact state (1 - contact, 0 - no contact), End-effector linear contact
                force (x,y,z).
        """
        if self._ee_name_to_idx is None:
            raise AttributeError(
                "`ee_names` attribute is not defined (is `None`). So this method cannot be used."
            )
        try:
            idx = self._ee_name_to_idx[ee_name]
        except KeyError as ke:
            raise KeyError(f"EE name '{ee_name}' not found in this object.") from ke
        else:
            ee_pose = None if self.ee_poses is None else self.ee_poses[idx]
            ee_twist = None if self.ee_twists is None else self.ee_twists[idx]
            contact_state = None if self.contact_states is None else self.contact_states[idx]
            contact_force = None if self.contact_forces is None else self.contact_forces[idx]
            return ee_pose, ee_twist, contact_state, contact_force

    def __getitem__(
        self, ee_name: str
    ) -> Tuple[Pose3D | None, Twist | None, int | None, Vector3D | None]:
        """Get the end-effector states (pose, contact state, contact force) of the specified
        end-effector frame.

        This method is equivalent to directly using the `get_state_of` method. I.e.
        The two lines below are equivalent:
            >> p, v, c, f = ee_states.get_state_of(ee_name)
            >> p, v, c, f = ee_states[ee_name]

        Args:
            ee_name (str): The name of the end-effector frame.

        Raises:
            AttributeError: Raised if `ee_names` attribute was not defined for this object.
            KeyError: If the specified `ee_name` is not in the `ee_names` of this object.

        Returns:
            Tuple[Pose3D | None, Twist | None, int | None, Vector3D | None]: End-effector pose,
                twist, contact state (1 - contact, 0 - no contact), End-effector linear contact
                force (x,y,z).
        """
        return self.get_state_of(ee_name=ee_name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "ee_names":
            super().__setattr__(
                "_ee_name_to_idx",
                (None if __value is None else {name: __value.index(name) for name in __value}),
            )
        elif __name in ["ee_poses", "ee_twists", "contact_states", "contact_forces"]:
            if __value is not None:
                if __name == "contact_states":
                    __value = np.array(__value)
                elif __name == "contact_forces":
                    __value = [np.array(v) if v is not None else None for v in __value]
        else:
            raise NameError(
                f"Tried to assign value to illegal parameter {__name} in object of type "
                f"{self.__class__.__name__}."
            )
        super().__setattr__(__name, __value)

    def extend(
        self,
        other: "EndEffectorStates",
        run_checks: bool = True,
        overwrite_existing: bool = False,
    ):
        """Extend the values of the attributes of this object using another instance of this class.

        Args:
            other (EndEffectorStates): The other instance whose values are to be used to extend
                this object.
            run_checks (bool, optional): If False, all values will be appended to existing ones
                for each attribute of this object, without checking for validity or repeats.
                Defaults to True.
            overwrite_existing (bool, optional): If set to True, will overwrite values with the
                new ones if same ee_name exists. Defaults to False.
        """
        if not (run_checks or overwrite_existing):
            self.ee_names = self.ee_names.copy() + other.ee_names
            self.ee_poses = self.ee_poses.copy() + other.ee_poses
            self.ee_twists = self.ee_twists.copy() + other.ee_twists
            self.contact_states = list(self.contact_states) + list(other.contact_states)
            self.contact_forces = list(self.contact_forces) + list(other.contact_forces)
        else:
            if not run_checks:
                logging.warning(
                    f"{self.__class__.__name__}: Setting `overwrite_existing` to True is"
                    " not allowed when `run_checks=False`. Will ignore the `run_checks`"
                    " flag for this now."
                )
            if not overwrite_existing:
                assert set(self.ee_names).isdisjoint(other.ee_names), (
                    f"{self.__class__.__name__}: EndEffectorStates object provided to extend this"
                    " class has ee names that are already present in current instance. To extend "
                    "this object by overwriting existing values for the common ee names, set "
                    "`override_estisting=True`"
                )
            ee_names = []
            ee_poses = []
            ee_twists = []
            ee_cs = []
            ee_forces = []
            for ee_name in self.ee_names + other.ee_names:
                if ee_name not in ee_names:
                    ee_names.append(ee_name)
                if ee_name in other.ee_names:
                    # if this ee is in the second object, info from that will be used,
                    # even if the same ee exists in the first object
                    p, v, c, f = other.get_state_of(ee_name=ee_name)
                else:
                    p, v, c, f = self.get_state_of(ee_name=ee_name)
                ee_poses.append(p)
                ee_twists.append(v)
                ee_cs.append(c)
                ee_forces.append(f)
            assert len(ee_names) == len(ee_poses) == len(ee_twists) == len(ee_cs)
            self.ee_names = ee_names
            self.ee_poses = ee_poses
            self.ee_twists = ee_twists
            self.contact_states = ee_cs
            self.contact_forces = ee_forces

    def update_from(self, other: "EndEffectorStates"):
        """Updates the attributes corresponding to each end-effector name of this object
        using values from another instance. Only the values of the end-effectors in the
        other instance will be modified; values of other end-effectors are left unchanged.

        Args:
            other (EndEffectorStates): The instance whose values will be used to modify
                the attributes of the current instance.
        """
        for n, j_name in enumerate(other.ee_names):
            if other.ee_poses is not None:
                self.ee_poses[self._ee_name_to_idx[j_name]] = other.ee_poses[n]
            if other.ee_twists is not None:
                self.ee_twists[self._ee_name_to_idx[j_name]] = other.ee_twists[n]
            if other.contact_states is not None:
                self.contact_states[self._ee_name_to_idx[j_name]] = other.contact_states[n]
            if other.contact_forces is not None:
                self.contact_forces[self._ee_name_to_idx[j_name]] = other.contact_forces[n]


@dataclass
class JointStates:
    """Joint state information/command/reference for an n-dof robot.
    All values are to be in the order specified in `joint_names`
    attribute."""

    joint_positions: np.ndarray = None
    joint_velocities: np.ndarray = None
    joint_efforts: np.ndarray = None
    joint_names: List[str] = None

    # internal mapping stored to make retrieval by name easier
    _joint_name_to_idx: Mapping[str, int] = field(default=None, init=False, repr=False)

    def get_state_of(self, joint_name: str) -> Tuple[float | None, float | None, float | None]:
        """Get the joint states (pos, vel, effort) of the specified joint.

        This method is equivalent to directly using the `__getitem__` of this object. I.e.
        The two lines below are equivalent:
            >> p, v, t = joint_states.get_state_of(joint_name)
            >> p, v, t = joint_states[joint_name]

        Args:
            joint_name (str): The name of the joint.

        Raises:
            AttributeError: Raised if `joint_names` attribute was not defined for this object.
            KeyError: If the specified `joint_name` is not in the `joint_names` of this object.

        Returns:
            Tuple[float | None, float | None, float | None]: pos, vel, effort of specified joint.
        """
        if self._joint_name_to_idx is None:
            raise AttributeError(
                "`joint_names` attribute is not defined (is `None`). So this method cannot be used."
            )
        try:
            idx = self._joint_name_to_idx[joint_name]
        except KeyError as ke:
            raise KeyError(f"Joint '{joint_name}' not found in this object.") from ke
        else:
            joint_position = None if self.joint_positions is None else self.joint_positions[idx]
            joint_velocity = None if self.joint_velocities is None else self.joint_velocities[idx]
            joint_effort = None if self.joint_efforts is None else self.joint_efforts[idx]
            return joint_position, joint_velocity, joint_effort

    def __getitem__(self, joint_name: str) -> Tuple[float | None, float | None, float | None]:
        """Get the joint states (pos, vel, effort) of the specified joint.

        This method is equivalent to directly using the `get_state_of` method of this class. I.e.
        The two lines below are equivalent:
            >> p, v, t = joint_states.get_state_of(joint_name)
            >> p, v, t = joint_states[joint_name]

        Args:
            joint_name (str): The name of the joint.

        Raises:
            AttributeError: Raised if `joint_names` attribute was not defined for this object.
            KeyError: If the specified `joint_name` is not in the `joint_names` of this object.

        Returns:
            Tuple[float | None, float | None, float | None]: pos, vel, effort of specified joint.
        """
        return self.get_state_of(joint_name=joint_name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "joint_names":
            super().__setattr__(
                "_joint_name_to_idx",
                (None if __value is None else {name: __value.index(name) for name in __value}),
            )
        elif __name in ["joint_positions", "joint_velocities", "joint_efforts"]:
            if __value is not None:
                __value = np.array(__value)
        else:
            raise NameError(
                f"Tried to assign value to illegal parameter {__name} in object ot type "
                f"{self.__class__.__name__}."
            )
        super().__setattr__(__name, __value)

    def extend(
        self,
        other: "JointStates",
        run_checks: bool = True,
        overwrite_existing: bool = False,
    ):
        """Extend the values of the attributes of this object using another instance of this class.

        Args:
            other (JointStates): The other instance whose values are to be used to extend
                this object.
            run_checks (bool, optional): If False, all values will be appended to existing ones
                for each attribute of this object, without checking for validity or repeats.
                Defaults to True.
            overwrite_existing (bool, optional): If set to True, will overwrite values with the
                new ones if same joint_name exists. Defaults to False.
        """
        if not (run_checks or overwrite_existing):
            self.joint_names = self.joint_names.copy() + other.joint_names
            self.joint_positions = np.append(self.joint_positions, other.joint_positions)
            self.joint_velocities = np.append(self.joint_velocities, other.joint_velocities)
            self.joint_efforts = np.append(self.joint_efforts, other.joint_efforts)
        else:
            if not run_checks:
                logging.warning(
                    f"{self.__class__.__name__}: Setting `overwrite_existing` to True is"
                    " not allowed when `run_checks=False`. Will ignore the `run_checks`"
                    " flag for this now."
                )
            if not overwrite_existing:
                assert set(self.joint_names).isdisjoint(other.joint_names), (
                    f"{self.__class__.__name__}: JointState object provided for extending this"
                    " class has joint names that are already present in current instance. To extend "
                    "this object by overwriting existing values for the common joints, set "
                    "`override_estisting=True`"
                )
            j_names = []
            j_pos = []
            j_vel = []
            j_eff = []
            for j_name in self.joint_names + other.joint_names:
                if j_name not in j_names:
                    j_names.append(j_name)
                if j_name in other.joint_names:
                    # if this joint is in the second object, info from that will be used,
                    # even if the same joint exists in the first object
                    p, v, t = other.get_state_of(joint_name=j_name)
                else:
                    p, v, t = self.get_state_of(joint_name=j_name)
                j_pos.append(p)
                j_vel.append(v)
                j_eff.append(t)
            assert len(j_names) == len(j_pos) == len(j_vel) == len(j_eff)
            self.joint_names = j_names
            self.joint_positions = j_pos
            self.joint_velocities = j_vel
            self.joint_efforts = j_eff

    def update_from(self, other: "JointStates"):
        """Updates the attributes corresponding to each joint name of this object
        using values from another instance. Only the values of the joints in the
        other instance will be modified; values of other joints are left unchanged.

        Args:
            other (JointStates): The instance whose values will be used to modify
                the attributes of the current instance.
        """
        for n, j_name in enumerate(other.joint_names):
            if other.joint_positions is not None:
                self.joint_positions[self._joint_name_to_idx[j_name]] = other.joint_positions[n]
            if other.joint_velocities is not None:
                self.joint_velocities[self._joint_name_to_idx[j_name]] = other.joint_velocities[n]
            if other.joint_efforts is not None:
                self.joint_efforts[self._joint_name_to_idx[j_name]] = other.joint_efforts[n]


@dataclass
class StateEstimates:
    """Possible outputs of a robot state estimator.

    This should contain state estimates that are computed and not typically observed
    directly using sensors.

    NOTE: It might be worth making base state estimates into type FrameMotion.
    """

    pose: Pose3D = field(default_factory=Pose3D)
    """Pose of the robot's base link in the world frame."""
    twist: Twist = field(default_factory=Twist)
    """Velocity of the robot in the base frame; linear and angular."""
    end_effector_states: EndEffectorStates = field(default_factory=EndEffectorStates)
    """State estimate data relating to end-effectors of this robot."""

    def extend(
        self,
        other: "StateEstimates",
        run_checks: bool = True,
        overwrite_existing: bool = False,
    ):
        """Extend the values of the `end_effector_states` attribute of this object using another
        instance of this class.

        Args:
            other (StateEstimates): The other instance whose values are to be used to extend
                this object.
            run_checks (bool, optional): If False, all values will be appended to existing ones
                for each attribute of `joint_states`, without checking for validity or repeats.
                Defaults to True.
            overwrite_existing (bool, optional): If set to True, will overwrite values with the
                new ones in `joint_states` if same joint_name exists. Defaults to False.
        """
        self.end_effector_states.extend(
            other=other.end_effector_states,
            run_checks=run_checks,
            overwrite_existing=overwrite_existing,
        )

    def update_from(self, other: "StateEstimates", update_base_state: bool):
        """Updates the attributes corresponding to each joint name of this object
        using values from another instance. Only the values of the joints in the
        other instance will be modified; values of other joints are left unchanged.

        Args:
            other (StateEstimates): The instance whose values will be used to modify
                the attributes of the current instance.
            update_base_state (bool): If set to True, will also modify the `pose` and
                `twist` attributes from the other instance.
        """
        if update_base_state:
            self.pose = other.pose
            self.twist = other.twist
        self.end_effector_states.update_from(other=other.end_effector_states)


@dataclass
class RobotState:
    """Represents the state of a robot.

    All joint values should be in the order mentioned in joint_names.
    """

    state_estimates: StateEstimates = field(default_factory=StateEstimates)
    """State estimator output."""
    joint_states: JointStates = field(default_factory=JointStates)
    """Proprioceptive data from the robot sensors/interfaces."""

    def extend(
        self,
        other: "RobotState",
        run_checks: bool = True,
        overwrite_existing: bool = False,
    ):
        """Extend the values of the `state_estimates` and `joint_states` attribute of this object
        using another instance of this class.

        Args:
            other (RobotState): The other instance whose values are to be used to extend
                this object.
            run_checks (bool, optional): If False, all values will be appended to existing ones
                for each attribute of `joint_states` and `state_estimates`, without checking for
                validity or repeats. Defaults to True.
            overwrite_existing (bool, optional): If set to True, will overwrite values with the
                new ones if same joint name of end-effector name exists in the other instance.
                Defaults to False.
        """
        self.state_estimates.extend(
            other=other.state_estimates,
            run_checks=run_checks,
            overwrite_existing=overwrite_existing,
        )
        self.joint_states.extend(
            other=other.joint_states,
            run_checks=run_checks,
            overwrite_existing=overwrite_existing,
        )

    def update_from(self, other: "RobotState", update_base_state: bool):
        """Updates the attributes corresponding to each joint name and end-effector name
        of this object using values from another instance. Only the values of the joints in the
        other instance will be modified; values of other joints are left unchanged.


        Args:
            other (RobotState): The instance whose values are to be used for modifying this
                object.
            update_base_state (bool): If set to True, will also modify the `pose` and
                `twist` attributes from the other instance's state_estimates.
        """
        self.state_estimates.update_from(
            other=other.state_estimates, update_base_state=update_base_state
        )
        self.joint_states.update_from(other=other.joint_states)


# pylint: disable=W0212
@dataclass
class RobotCmd:
    """Object containing low-level commands to be sent directly to the robot.

    This is computed by the controller. All joint orders should be same as the order in RobotState.

    This object follows the PVT-PD (position, velocity, effort, stiffness (P), damping (D))
    actuation command protocol (but for sequence of actuators, instead of just one).

    Attributes (all values have to be filled in the order defined in `joint_commands.joint_names`):
        joint_commands.joint_names (List[str])
        joint_commands.joint_positions (np.ndarray)
        joint_commands.joint_velocities (np.ndarray)
        joint_commands.joint_efforts (np.ndarray)
        Kp (np.ndarray)
        Kd (np.ndarray)
    """

    joint_commands: JointStates = field(default_factory=JointStates)
    Kp: np.ndarray = None
    """Array of joint position gains for each joint (stiffness) in `joint_commands.joint_names`
    order."""
    Kd: np.ndarray = None
    """Array of joint velocity gains (or damping if `joint_velocities=None` or zeros) in 
    `joint_commands.joint_names` order."""

    def set_joint_gains(self, joint_names: List[str], Kp: np.ndarray = None, Kd: np.ndarray = None):
        """Set Kp and/or Kd gains for a list of specified joints.

        Args:
            joint_names (List[str]): Names of joints whose gains are to be modified.
            Kp (np.ndarray, optional): New Kp value (per specified joint). Defaults to None (no
                change).
            Kd (np.ndarray, optional): New Kd value (per specified joint). Defaults to None (no
                change).
        """
        assert (
            self.joint_commands.joint_names is not None
        ), "Joint names not provided in `joint_commands.joint_names`."
        if self.Kp is None:
            self.Kp = np.zeros(len(self.joint_commands.joint_names))
        if self.Kd is None:
            self.Kd = np.zeros(len(self.joint_commands.joint_names))
        for n, jname in enumerate(joint_names):
            if Kp is not None:
                self.Kp[self.joint_commands._joint_name_to_idx[jname]] = Kp[n]
            if Kd is not None:
                self.Kd[self.joint_commands._joint_name_to_idx[jname]] = Kd[n]

    def get_command_for_joint(self, joint_name: str) -> Tuple[float, float, float, float, float]:
        """Get the commands for the specified joint.

        Args:
            joint_name (str): Name of the joint.

        Returns:
            Tuple[float, float, float, float, float]: Kp, Kd, position, velocity, effort
        """
        jid = self.joint_commands._joint_name_to_idx[joint_name]
        return (
            self.Kp[jid],
            self.Kd[jid],
            self.joint_commands.joint_positions[jid],
            self.joint_commands.joint_velocities[jid],
            self.joint_commands.joint_efforts[jid],
        )

    @classmethod
    def createZeros(cls: "RobotCmd", dof: int, joint_names: List[str] = None) -> "RobotCmd":
        """Static class method to create a pre-populated RobotCmd with `dof` number of joints.

        All values are set to zeros (including gains).

        Args:
            dof (int): Number of joints. Values will be set to numpy arrays of this size.
            joint_names (List[float], optional): List of joint names. By default, this is None,
                and will have to be filled later if needed.

        Returns:
            RobotCmd: An instance of RobotCmd with all values zero with the specified number of
                joints.
        """
        assert isinstance(dof, int)
        if joint_names is not None:
            assert len(joint_names) == dof
        return cls(
            joint_commands=JointStates(
                joint_positions=np.zeros(dof),
                joint_velocities=np.zeros(dof),
                joint_efforts=np.zeros(dof),
                joint_names=copy.deepcopy(joint_names),
            ),
            Kp=np.zeros(dof),
            Kd=np.zeros(dof),
        )

    @classmethod
    def fromJointStates(
        cls: "RobotCmd",
        joint_states: JointStates,
        Kp: float | np.ndarray,
        Kd: float | np.ndarray,
    ) -> "RobotCmd":
        """Static class method to create a RobotCmd object using a provided JointStates object.

        This method will directly use values from the provided JointStates object and set them
        as the values for `joint_commands`. Kp and Kd are read from the arguments.

        Args:
            joint_states (JointStates): The JointStates object whose values are to be used.
            Kp (float | np.ndarray): Kp for position tracking.
            Kd (float | np.ndarray): Kd for velocity tracking (or damping)

        Returns:
            RobotCmd: An instance of RobotCmd with all values from the provided JointStates object.
        """
        return cls(joint_commands=copy.deepcopy(joint_states), Kp=Kp, Kd=Kd)

    def __setattr__(self, prop, val):
        # custom handling to make sure kp and kd dimensions and signs are valid
        if prop in ["Kp", "Kd"] and val is not None:
            if prop == "Kp":
                _compare_with = self.joint_commands.joint_positions
            else:
                _compare_with = self.joint_commands.joint_velocities
            if _compare_with is not None:
                if isinstance(val, Number):
                    val = [val] * len(_compare_with)
                else:
                    if len(val) == 1:
                        val = [val[0]] * len(_compare_with)
                    assert len(val) == len(
                        _compare_with
                    ), f"{prop} dimensions ({len(val)}) should match dimension of {'joint_positions' if prop == 'Kp' else 'joint_velocities'} ({len(_compare_with)}) in robot cmd."
            assert np.all(
                val == np.abs(val)
            ), f"All values for {prop} should be positive. Invalid value provided: {val}"
            val = np.array(val)
        super().__setattr__(prop, val)

    def extend(
        self,
        other: "RobotCmd",
        run_checks: bool = True,
        overwrite_existing: bool = False,
    ):
        """Extend the values of the attributes of this object using another instance of this class.

        Args:
            other (RobotCmd): The other instance whose values are to be used to extend
                this object.
            run_checks (bool, optional): If False, all values will be appended to existing ones
                for each attribute of this object, without checking for validity or repeats.
                Defaults to True.
            overwrite_existing (bool, optional): If set to True, will overwrite values with the
                new ones if same joint_name exists. Defaults to False.
        """
        new_kps = np.append(self.Kp, other.Kp)
        new_kds = np.append(self.Kd, other.Kd)
        j_names = self.joint_commands.joint_names + other.joint_commands.joint_names
        self.joint_commands.extend(
            other=other.joint_commands,
            run_checks=run_checks,
            overwrite_existing=overwrite_existing,
        )
        if not (run_checks or overwrite_existing):
            self.Kp = new_kps
            self.Kd = new_kds
        else:
            self.set_joint_gains(joint_names=j_names, Kp=new_kps, Kd=new_kds)

    def update_from(self, other: "RobotCmd"):
        """Updates the attributes corresponding to each joint name of this object
        using values from another instance. Only the values of the joints in the
        other instance will be modified; values of joints not in the other instance
        are left unchanged.

        Args:
            other (RobotCmd): The instance whose values will be used to modify
                the attributes of the current instance.
        """
        for n, jname in enumerate(other.joint_commands.joint_names):
            self.Kp[self.joint_commands._joint_name_to_idx[jname]] = other.Kp[n]
            self.Kd[self.joint_commands._joint_name_to_idx[jname]] = other.Kd[n]
            # more efficient to update all joint states and gains here in the same loop than
            # to call `self.joint_commands.update_from(other=other.joint_commands)`
            if other.joint_commands.joint_positions is not None:
                self.joint_commands.joint_positions[
                    self.joint_commands._joint_name_to_idx[jname]
                ] = other.joint_commands.joint_positions[n]
            if other.joint_commands.joint_velocities is not None:
                self.joint_commands.joint_velocities[
                    self.joint_commands._joint_name_to_idx[jname]
                ] = other.joint_commands.joint_velocities[n]
            if other.joint_commands.joint_efforts is not None:
                self.joint_commands.joint_efforts[self.joint_commands._joint_name_to_idx[jname]] = (
                    other.joint_commands.joint_efforts[n]
                )
