import copy
from typing import List, Mapping
import numpy as np
import pybullet_robot
from pybullet_robot.utils.robot_loader_utils import get_urdf_from_awesome_robot_descriptions

from .sim_robot_base import SimulatedRobotInterface
from ....core.types import (
    Pose3D,
    RobotState,
    StateEstimates,
    RobotCmd,
    Twist,
    EndEffectorStates,
    QuatType,
    JointStates,
)
from ....utils.math_utils import quat2rot
from ....utils.time_utils import ClockBase
from ....core.logging import logging
from ....utils.urdf_utils import (
    temp_path_urdf_with_joints_fixed,
    create_new_urdf_with_joints_fixed,
)

# pylint: disable = C0103


class PybulletRobot(SimulatedRobotInterface):
    """A RobotInterface implementation for a generic robot in pybullet.

    All subclasses of this method will use synchronous sim time stepping, i.e.
    the simulated world will step only when the read() method of this class is
    called. This is to keep the clock synchronous and for providing full repeatable
    performance to the control loop by minimising parallelisation.
    """

    DEFAULT_URDF_PATH: str = None
    """Default urdf path to be used for particular instances of PybulletRobot. This
    variable should be overridden in the implemented child classes."""

    class BulletWorldClock(ClockBase):
        """Only valid if sim stepping is done synchronously and managed by the PybulletRobot class."""

        def __init__(self, dt: float) -> None:
            self._current_t = 0.0
            self._dt = dt

        def step_time(self):
            """To be called each time simulation steps forward."""
            self._current_t += self._dt

        def get_time(self) -> float:
            return self._current_t

    def __init__(
        self,
        urdf_path: str,
        ee_names: List[str] = None,
        place_on_ground: bool = True,
        default_base_position: np.ndarray = np.zeros(3),
        default_base_orientation: np.ndarray = np.array([0, 0, 0, 1]),
        default_joint_positions: np.ndarray = None,
        create_pinocchio_interface: bool = True,
        floating_base: bool = True,
        verbose: bool = True,
        **sim_interface_kwargs,
    ):
        """A RobotInterface implementation for a generic robot in pybullet.

        All subclasses of this method will use synchronous sim time stepping, i.e.
        the simulated world will step only when the read() method of this class is
        called. This is to keep the clock synchronous and for providing full repeatable
        performance to the control loop by minimising parallelisation.

        Args:
            urdf_path (str): Path to urdf file of robot.
            ee_names (List[str], optional): List of end-effectors for the robot. Defaults to None.
            place_on_ground (bool): If true, the base position height will automatically be
                adjusted such that the robot is on the ground with the given joint positions.
                Defaults to True.
            default_base_position (np.ndarray, optional): Default position of the base of the robot
                in the world frame during start-up. Note that the height value (z) is only used if
                'place_on_ground' is set to False. Defaults to np.zeros(3).
            default_base_orientation (np.ndarray, optional): Default orientation quaternion in the
                world frame for the base of the robot during start-up. Defaults to
                np.array([0, 0, 0, 1]).
            default_joint_positions (List[float], optional): Optional starting values for joints.
                Defaults to None. NOTE: These values should be in the order of joints in pybullet.
            create_pinocchio_interface (bool, optional): If true, creates a PinocchioInterface
                object for this robot. The object is automatically updated with latest state when
                read() is called. Defaults to True.
            floating_base (bool, optional): Specifying whether this robot has a floating or fixed
                base (setting to True creates a floating base instance with pinocchio, and makes
                the simulated robot free floating (not fixed base)). Defaults to True.
            verbose (bool, optional): Verbosity flag for debugging robot info during construction.
                Defaults to True.

            **sim_interface_kwargs: additional keyword arguments to pass down to the internal
                Simulator interface (i.e. additional arguments to pybullet_robot.BulletRobot).
        """
        if urdf_path is None:
            urdf_path = self.DEFAULT_URDF_PATH

        sim_interface_kwargs["enable_torque_mode"] = sim_interface_kwargs.get(
            "enable_torque_mode", True
        )

        if ee_names is None:
            ee_names = []

        self._sim_robot = pybullet_robot.BulletRobot(
            urdf_path=urdf_path,
            run_async=False,  # this interface will only allows synchronous sim stepping
            ee_names=ee_names,
            default_joint_positions=default_joint_positions,
            place_on_ground=place_on_ground,
            default_base_position=default_base_position,
            default_base_orientation=default_base_orientation,
            verbose=verbose,
            use_fixed_base=(not floating_base),
            **sim_interface_kwargs,
        )

        # this is only usable because we use synchronous time stepping in sim
        self._clock = self.BulletWorldClock(
            dt=self._sim_robot.get_physics_parameters()["fixedTimeStep"]
        )

        super().__init__(
            robot_urdf=urdf_path,
            ee_names=ee_names,
            floating_base=floating_base,
            verbose=verbose,
            create_pinocchio_interface=create_pinocchio_interface,
        )
        if self.has_pinocchio_interface:
            sim_robot_state = copy.deepcopy(self._sim_robot.get_robot_states())
            rot_mat = quat2rot(quaternion=sim_robot_state["base_quaternion"]).T
            state = RobotState(
                joint_states=JointStates(
                    joint_names=self._sim_robot.actuated_joint_names,
                    joint_positions=sim_robot_state["actuated_joint_positions"],
                    joint_velocities=sim_robot_state["actuated_joint_velocities"],
                    joint_efforts=sim_robot_state["actuated_joint_torques"],
                ),
                state_estimates=StateEstimates(
                    pose=Pose3D(
                        position=sim_robot_state["base_position"],
                        orientation=sim_robot_state["base_quaternion"],
                    ),
                    twist=Twist(  # twist represented in base frame
                        linear=rot_mat @ sim_robot_state["base_velocity_linear"],
                        angular=rot_mat @ sim_robot_state["base_velocity_angular"],
                    ),
                    end_effector_states=EndEffectorStates(
                        contact_states=sim_robot_state["ee_contact_states"],
                        ee_names=sim_robot_state["ee_order"],
                    ),
                ),
            )
            self.update_pinocchio_robot_state(robot_state=state)

    def read(self) -> RobotState:
        self._sim_robot.step()  # synchronised sim stepping
        self._clock.step_time()

        sim_robot_state = copy.deepcopy(self._sim_robot.get_robot_states())

        try:
            rot_mat = quat2rot(quaternion=sim_robot_state["base_quaternion"]).T
        except ValueError:
            print(f"Sim quaternion: {sim_robot_state['base_quaternion']}")
            raise

        state = RobotState(
            joint_states=JointStates(
                joint_names=self._sim_robot.actuated_joint_names,
                joint_positions=sim_robot_state["actuated_joint_positions"],
                joint_velocities=sim_robot_state["actuated_joint_velocities"],
                joint_efforts=sim_robot_state["actuated_joint_torques"],
            ),
            state_estimates=StateEstimates(
                pose=Pose3D(
                    position=sim_robot_state["base_position"],
                    orientation=sim_robot_state["base_quaternion"],
                ),
                twist=Twist(  # twist represented in base frame
                    linear=rot_mat @ sim_robot_state["base_velocity_linear"],
                    angular=rot_mat @ sim_robot_state["base_velocity_angular"],
                ),
                end_effector_states=EndEffectorStates(
                    contact_states=sim_robot_state["ee_contact_states"],
                    ee_names=sim_robot_state["ee_order"],
                ),
            ),
        )

        if self.has_pinocchio_interface:
            state = self.update_pinocchio_robot_state(robot_state=state, update_ee_states=True)

        return state

    def write(self, cmd: RobotCmd) -> bool:
        if (jnames := cmd.joint_commands.joint_names) is None:
            return True
        try:
            self._sim_robot.set_actuated_joint_commands(
                actuated_joint_names=(jnames),
                q=(
                    cmd.joint_commands.joint_positions
                    if cmd.joint_commands.joint_positions is not None
                    else np.zeros(len(jnames))
                ),
                Kp=cmd.Kp if cmd.Kp is not None else np.zeros(len(jnames)),
                dq=(
                    cmd.joint_commands.joint_velocities
                    if cmd.joint_commands.joint_velocities is not None
                    else np.zeros(len(jnames))
                ),
                Kd=cmd.Kd if cmd.Kd is not None else np.zeros(len(jnames)),
                tau=(
                    cmd.joint_commands.joint_efforts
                    if cmd.joint_commands.joint_efforts is not None
                    else np.zeros(len(jnames))
                ),
            )
            return True
        except Exception as e:  # pylint:disable=W0718
            logging.exception(
                f"{__class__.__name__}: Control command could not be written to simulator. {e}"
                f" Culprit {cmd}. len joints: {len(self._sim_robot.actuated_joint_names)}."
            )
            return False

    def set_base_pose(self, position: np.ndarray, orientation: QuatType):
        self._sim_robot.reset_base_pose(position=position, orientation=orientation)

    def get_sim_clock(self) -> ClockBase:
        return self._clock

    def shutdown(self):
        super().shutdown()
        if hasattr(self, "_sim_robot"):
            self._sim_robot.shutdown()

    @classmethod
    def withJointsLocked(
        cls: "PybulletRobot",
        joints_to_lock: Mapping[str, float] = None,
        temp_urdf: bool = True,
        **kwargs,
    ):
        """Classmethod for RobotInterface implementation for PybulletRobot robot in pybullet, that
        locks the specified joints of the robot. All specified joints are converted to "fixed"
        joint type and cannot be reverted dynamically!

        NOTE: Use this method with caution. All keyword arguments are directly passed to the
        default constructor!

        Args:
            joints_to_lock (Mapping[str, float]): Dictionary containing name of joints that should
                be locked as keys, and the joint values to lock them at as values. By default, uses
                upper body joints (from upper torso upwards) set to zeros.
            temp_urdf (bool, optional): If set to True, will create a temporary urdf that will be
                deleted after loading in pybullet, otherwise create a file in /tmp/ directory.
                Defaults to True.

            **kwargs: additional keyword arguments to pass to PybulletRobot or subclass.


        -------------------------------------------------------------------------------------------
        NOTE
        ----
        By default, the urdf created via this constructor is only valid during construction. The
        path to the urdf (accessed through self._sim_robot.urdf_path, for instance) will not be a
        valid path once the init is completed. This is because this method temporarily creates a
        urdf after modifying the specified joints to "fixed" joint type, and destroys the temporary
        file soon after all objects using it (pybullet, pinocchio, etc.) are created in the
        construction.

        To disable this behaviour, set `temp_urdf=False` during construction. This will create a
        permenant urdf file in the /tmp/ directory.
        -------------------------------------------------------------------------------------------
        """
        if joints_to_lock is None:
            joints_to_lock = {}

        urdf_path = kwargs.get("urdf_path", cls.DEFAULT_URDF_PATH)
        kwargs.pop("urdf_path", None)
        if temp_urdf:
            with temp_path_urdf_with_joints_fixed(urdf_path, joints_to_lock) as updated_urdf:
                return cls(urdf_path=updated_urdf, **kwargs)

        updated_urdf = create_new_urdf_with_joints_fixed(urdf_path, joints_to_lock)
        return cls(urdf_path=updated_urdf, **kwargs)

    @classmethod
    def fromAwesomeRobotDescriptions(
        cls: "PybulletRobot",
        robot_description_name: str,
        ee_names: List[str] = None,
        place_on_ground: bool = True,
        default_base_position: np.ndarray = np.zeros(3),
        default_base_orientation: np.ndarray = np.array([0, 0, 0, 1]),
        default_joint_positions: np.ndarray = None,
        create_pinocchio_interface: bool = True,
        floating_base: bool = True,
        verbose: bool = True,
        **sim_interface_kwargs,
    ) -> "PybulletRobot":
        """Create a PybulletRobot instance using robots available in the open source Awesome Robot
        Descriptions list
        (https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions).


        The list of available robot descriptions can also be viewed in the variable `AWESOME_ROBOTS`
        imported from `pybullet_robot.utils.urdf_utils`.

        Downloads description package for the specified robot and caches it locally (only needs
        downloading once), and loads this robot as a PybulletRobot instance. NOTE: `ee_names` should
        be provided if proper end-effector kinematics is required. Also starting pose and joint
        configurations are always at zero. So to set to some specific configuration provide
        appropriate arguments.

        Args:
            robot_description_name (str): The name of the robot description package name. Should be
                a value available in the awesome robots list (also defined in variable
                `AVAILABLE_ROBOTS` in `utils.urdf_utils`).
            ee_names (List[str], optional): List of end-effectors for the robot. Defaults to None.
            place_on_ground (bool): If true, the base position height will automatically be adjusted
                such that the robot is on the ground with the given joint positions. Defaults to
                True.
            default_base_position (np.ndarray, optional): Default position of the base of the robot
                in the world frame during start-up. Note that the height value (z) is only used if
                'place_on_ground' is set to False. Defaults to np.zeros(3).
            default_base_orientation (np.ndarray, optional): Default orientation quaternion in the
                world frame for the base of the robot during start-up. Defaults to
                np.array([0, 0, 0, 1]).
            default_joint_positions (List[float], optional): Optional starting values for joints.
                Defaults to None. NOTE: These values should be in the order of joints in pybullet.
            create_pinocchio_interface (bool, optional): If true, creates a PinocchioInterface
                object for this robot. The object is automatically updated with latest state when
                read() is called. Defaults to True.
            floating_base (bool, optional): Specifying whether this robot has a floating or fixed
                base (setting to True creates a floating base instance with pinocchio, and makes the
                simulated robot free floating (not fixed base)). Defaults to True.
            verbose (bool, optional): Verbosity flag for debugging robot info during construction.
                Defaults to True.

        Returns:
            PybulletRobot: A PybulletRobot instance created using the URDF generated using the awesome
                robot descriptions package.
        """
        return cls(
            urdf_path=get_urdf_from_awesome_robot_descriptions(
                robot_description_pkg_name=robot_description_name
            ),
            ee_names=ee_names,
            place_on_ground=place_on_ground,
            default_base_position=default_base_position,
            default_base_orientation=default_base_orientation,
            default_joint_positions=default_joint_positions,
            create_pinocchio_interface=create_pinocchio_interface,
            floating_base=floating_base,
            verbose=verbose,
            **sim_interface_kwargs,
        )
