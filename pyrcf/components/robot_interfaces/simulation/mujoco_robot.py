"""A RobotInterface implementation for a generic robot in MuJoCo.

This mirrors `PybulletRobot` so that the same control loop and the same components can be run
against either simulator, which makes sim-to-sim differences observable (and any pyrcf-side bug
distinguishable from a physics difference).
"""

from typing import List

import numpy as np

from .sim_robot_base import SimulatedRobotInterface
from ....core.logging import logger
from ....core.types import (
    EndEffectorStates,
    JointStates,
    Pose3D,
    QuatType,
    RobotCmd,
    RobotState,
    StateEstimates,
    Twist,
)
from ....utils.math_utils import quat2rot
from ....utils.time_utils import ClockBase
from ....variables import MUJOCO_ROBOT_AVAILABLE

if MUJOCO_ROBOT_AVAILABLE:
    import mujoco_robot  # pylint: disable=E0401
    from mujoco_robot.utils.robot_loader_utils import (  # pylint: disable=E0401
        get_mjcf_from_awesome_robot_descriptions,
        get_urdf_from_awesome_robot_descriptions,
    )

# NOTE: `mujoco-robot` is an optional dependency, so it is imported conditionally above. Every code
# path that touches it is reached only after the `MUJOCO_ROBOT_AVAILABLE` check in `__init__` has
# raised, but static analysis cannot follow that, hence the module-level suppression (same pattern
# as `torchscript_agent_base.py`).
# pylint: disable = C0103, possibly-used-before-assignment


class MujocoRobot(SimulatedRobotInterface):
    """A RobotInterface implementation for a generic robot in MuJoCo.

    All subclasses of this method will use synchronous sim time stepping, i.e.
    the simulated world will step only when the read() method of this class is
    called. This is to keep the clock synchronous and for providing full repeatable
    performance to the control loop by minimising parallelisation.

    NOTE: MuJoCo physics is described by an MJCF model, but `PinocchioInterface` can only be built
    from a URDF (pinocchio < 3 cannot parse MJCF). When loading from an MJCF, pass
    `pinocchio_urdf_path` to get a pinocchio interface, and be aware that the two model
    descriptions are not guaranteed to be identical.
    """

    DEFAULT_URDF_PATH: str = None
    """Default urdf path to be used for particular instances of MujocoRobot. This
    variable should be overridden in the implemented child classes."""

    DEFAULT_MJCF_PATH: str = None
    """Default MJCF (mujoco xml) path to be used for particular instances of MujocoRobot. This
    variable should be overridden in the implemented child classes."""

    class MujocoWorldClock(ClockBase):
        """Only valid if sim stepping is done synchronously and managed by the MujocoRobot class."""

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
        urdf_path: str = None,
        mjcf_path: str = None,
        ee_names: List[str] = None,
        place_on_ground: bool = True,
        default_base_position: np.ndarray = None,
        default_base_orientation: np.ndarray = None,
        default_joint_positions: np.ndarray = None,
        create_pinocchio_interface: bool = True,
        pinocchio_urdf_path: str = None,
        pinocchio_ee_names: List[str] = None,
        floating_base: bool = True,
        render: bool = True,
        read_contact_forces: bool = True,
        verbose: bool = True,
        **sim_interface_kwargs,
    ):
        """A RobotInterface implementation for a generic robot in MuJoCo.

        All subclasses of this method will use synchronous sim time stepping, i.e.
        the simulated world will step only when the read() method of this class is
        called.

        Args:
            urdf_path (str, optional): Path to urdf file of robot. Either this or `mjcf_path` has
                to be provided (or set as the class defaults). Defaults to None.
            mjcf_path (str, optional): Path to MJCF (mujoco xml) file of the robot. Takes
                precedence over `urdf_path` for the physics model. Defaults to None.
            ee_names (List[str], optional): List of end-effectors for the robot. Defaults to None.
            place_on_ground (bool): If true, the base position height will automatically be
                adjusted such that the robot is on the ground with the given joint positions.
                Defaults to True.
            default_base_position (np.ndarray, optional): Default position of the base of the robot
                in the world frame during start-up. Note that the height value (z) is only used if
                'place_on_ground' is set to False. Defaults to None (np.zeros(3)).
            default_base_orientation (np.ndarray, optional): Default orientation quaternion in the
                world frame for the base of the robot during start-up. Defaults to None
                (np.array([0, 0, 0, 1])).
            default_joint_positions (List[float], optional): Optional starting values for joints.
                Defaults to None. NOTE: These values should be in the order of the robot's
                actuated joints.
            create_pinocchio_interface (bool, optional): If true, creates a PinocchioInterface
                object for this robot. The object is automatically updated with latest state when
                read() is called. Defaults to True.
            pinocchio_urdf_path (str, optional): URDF to use for the PinocchioInterface. Only
                needed when the physics model is loaded from an MJCF, because pinocchio cannot
                parse MJCF. Defaults to None (use `urdf_path`).
            pinocchio_ee_names (List[str], optional): End-effector frame names as they appear in the
                URDF, when these differ from the body names in the MJCF. For example MuJoCo
                Menagerie's go2 has no `*_foot` bodies (the feet are geoms on `*_calf`), while the
                URDF does, so the sim needs `FL_calf` and pinocchio needs `FL_foot`. Defaults to
                None (use `ee_names` for both).
            floating_base (bool, optional): Specifying whether this robot has a floating or fixed
                base (setting to True creates a floating base instance with pinocchio, and makes
                the simulated robot free floating (not fixed base)). Defaults to True.
            render (bool, optional): If True, launches the MuJoCo passive viewer (needs a display).
                Set to False for headless use. Defaults to True.
            read_contact_forces (bool, optional): If True, fills
                `state_estimates.end_effector_states.contact_forces` on every `read()` with the net
                world-frame contact force on each end-effector. Set to False to skip the (small)
                per-contact computation. Defaults to True.
            verbose (bool, optional): Verbosity flag for debugging robot info during construction.
                Defaults to True.

            **sim_interface_kwargs: additional keyword arguments to pass down to the internal
                Simulator interface (i.e. additional arguments to mujoco_robot.MujocoRobot).

        Raises:
            RuntimeError: If the `mujoco-robot` package is not installed.
            ValueError: If neither a urdf nor an mjcf path is available.
        """
        if not MUJOCO_ROBOT_AVAILABLE:
            raise RuntimeError(
                f"The `mujoco-robot` package has to be installed for the {self.__class__.__name__}"
                " class to be used. e.g. `pip install pyrcf[mujoco]`."
            )

        if urdf_path is None:
            urdf_path = self.DEFAULT_URDF_PATH
        if mjcf_path is None:
            mjcf_path = self.DEFAULT_MJCF_PATH

        if urdf_path is None and mjcf_path is None:
            raise ValueError(
                f"{self.__class__.__name__}: One of `urdf_path` or `mjcf_path` has to be provided."
            )

        if default_base_position is None:
            default_base_position = np.zeros(3)

        if default_base_orientation is None:
            default_base_orientation = np.array([0, 0, 0, 1])

        sim_interface_kwargs["enable_torque_mode"] = sim_interface_kwargs.get(
            "enable_torque_mode", True
        )

        if ee_names is None:
            ee_names = []

        self._read_contact_forces = read_contact_forces

        # `pinocchio_ee_names` must be given in the same order as `ee_names`: state read from the
        # simulator is indexed by the MJCF names, then `update_pinocchio_robot_state` relabels it
        # with the URDF names, so the two lists are matched positionally.
        if pinocchio_ee_names is not None and len(pinocchio_ee_names) != len(ee_names):
            raise ValueError(
                f"{self.__class__.__name__}: `pinocchio_ee_names` ({len(pinocchio_ee_names)}) must"
                f" have the same length as `ee_names` ({len(ee_names)}), and be in the same order,"
                " because end-effector state is matched between the two models positionally."
            )

        self._sim_robot = mujoco_robot.MujocoRobot(
            urdf_path=urdf_path,
            mjcf_path=mjcf_path,
            run_async=False,  # this interface only allows synchronous sim stepping
            ee_names=ee_names,
            default_joint_positions=default_joint_positions,
            place_on_ground=place_on_ground,
            default_base_position=default_base_position,
            default_base_orientation=default_base_orientation,
            verbose=verbose,
            use_fixed_base=(not floating_base),
            render=render,
            **sim_interface_kwargs,
        )

        # this is only usable because we use synchronous time stepping in sim
        self._clock = self.MujocoWorldClock(dt=self._sim_robot.get_timestep())

        # pinocchio can only be built from a urdf; when the physics model came from an mjcf, the
        # caller has to say which urdf describes the same robot
        pin_urdf = pinocchio_urdf_path if pinocchio_urdf_path is not None else urdf_path
        if create_pinocchio_interface and pin_urdf is None:
            logger.warning(
                f"{self.__class__.__name__}: A PinocchioInterface was requested but this robot was"
                " loaded from an MJCF and no `pinocchio_urdf_path` was given (pinocchio cannot"
                " parse MJCF). Continuing without a pinocchio interface; controllers that need it"
                " will not work."
            )
        elif create_pinocchio_interface and mjcf_path is not None:
            logger.warning(
                f"{self.__class__.__name__}: Physics is simulated from '{mjcf_path}' but the"
                f" PinocchioInterface is built from '{pin_urdf}'. These are separate model"
                " descriptions and are not guaranteed to have identical kinematics/inertias."
            )

        super().__init__(
            robot_urdf=pin_urdf,
            ee_names=(pinocchio_ee_names if pinocchio_ee_names is not None else ee_names),
            floating_base=floating_base,
            verbose=verbose,
            create_pinocchio_interface=create_pinocchio_interface,
        )

        if self.has_pinocchio_interface:
            self._assert_pinocchio_joints_match()
            self.update_pinocchio_robot_state(robot_state=self._read_sim_state())

    def _assert_pinocchio_joints_match(self):
        """Check the simulated model and the pinocchio model agree on actuated joint names.

        When physics comes from an MJCF and pinocchio from a URDF, the two descriptions of the same
        hardware often use different joint naming conventions (e.g. MuJoCo Menagerie's `joint1` vs
        example-robot-data's `iiwa_joint_1`). Without this check the mismatch surfaces much later
        as an opaque `KeyError` from inside `PinocchioInterface.update()`.

        Raises:
            ValueError: If the joint names do not match.
        """
        sim_joints = list(self._sim_robot.actuated_joint_names)
        pin_joints = list(self.get_pinocchio_interface().actuated_joint_names)
        if set(sim_joints) == set(pin_joints):
            return

        missing = [j for j in sim_joints if j not in set(pin_joints)]
        raise ValueError(
            f"{self.__class__.__name__}: the simulated model and the pinocchio model do not agree"
            f" on actuated joint names, so state cannot be mapped between them.\n"
            f"  simulated ({len(sim_joints)}): {sim_joints[:4]}{' ...' if len(sim_joints) > 4 else ''}\n"
            f"  pinocchio ({len(pin_joints)}): {pin_joints[:4]}{' ...' if len(pin_joints) > 4 else ''}\n"
            f"  {len(missing)} simulated joint(s) absent from the pinocchio model.\n"
            "This usually means the MJCF and the URDF use different naming conventions for the"
            " same robot. Either use a robot whose descriptions agree (e.g. go2, g1, anymal_c),"
            " load the physics model from the URDF instead of the MJCF, or pass"
            " `create_pinocchio_interface=False` if the components in use do not need it."
        )

    def _read_sim_state(self) -> RobotState:
        """Build a `RobotState` from the current simulator state (without stepping).

        Returns:
            RobotState: the latest state of the robot.
        """
        # NOTE: unlike the pybullet interface, no defensive deepcopy is needed here --
        # `get_robot_states()` allocates fresh arrays rather than exposing internal buffers.
        sim_state = self._sim_robot.get_robot_states()

        # twist is reported in the world frame by the simulator; the pyrcf contract is base frame
        rot_mat = quat2rot(quaternion=sim_state.base_quaternion).T

        # NOTE: `EndEffectorStates.contact_forces` is part of the pyrcf state contract but the
        # pybullet interface never fills it in. MuJoCo reports per-contact forces, so populate it
        # here -- this is what makes force/admittance control usable on this backend.
        contact_forces = None
        if self._read_contact_forces and sim_state.ee_order:
            contact_forces = [
                self._sim_robot.get_link_contact_force(name) for name in sim_state.ee_order
            ]

        return RobotState(
            joint_states=JointStates(
                joint_names=sim_state.joint_order,
                joint_positions=sim_state.actuated_joint_positions,
                joint_velocities=sim_state.actuated_joint_velocities,
                joint_efforts=sim_state.actuated_joint_torques,
            ),
            state_estimates=StateEstimates(
                pose=Pose3D(
                    position=sim_state.base_position,
                    orientation=sim_state.base_quaternion,
                ),
                twist=Twist(  # twist represented in base frame
                    linear=rot_mat @ sim_state.base_velocity_linear,
                    angular=rot_mat @ sim_state.base_velocity_angular,
                ),
                end_effector_states=EndEffectorStates(
                    contact_states=sim_state.ee_contact_states,
                    contact_forces=contact_forces,
                    ee_names=sim_state.ee_order,
                ),
            ),
        )

    def read(self) -> RobotState:
        self._sim_robot.step()  # synchronised sim stepping
        self._clock.step_time()

        state = self._read_sim_state()

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
            logger.exception(
                f"{self.__class__.__name__}: Control command could not be written to simulator."
                f" {e} Culprit {cmd}. len joints:"
                f" {len(self._sim_robot.actuated_joint_names)}."
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
    def fromAwesomeRobotDescriptions(
        cls: "MujocoRobot",
        robot_description_name: str,
        ee_names: List[str] = None,
        place_on_ground: bool = True,
        default_base_position: np.ndarray = None,
        default_base_orientation: np.ndarray = None,
        default_joint_positions: np.ndarray = None,
        create_pinocchio_interface: bool = True,
        pinocchio_urdf_description: str = None,
        pinocchio_ee_names: List[str] = None,
        floating_base: bool = True,
        render: bool = True,
        verbose: bool = True,
        **sim_interface_kwargs,
    ) -> "MujocoRobot":
        """Create a MujocoRobot instance using robots available in the open source Awesome Robot
        Descriptions list
        (https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions).

        Downloads the description package for the specified robot and caches it locally (only needs
        downloading once).

        A description name ending in `_mj_description` is loaded as an MJCF (these come from
        MuJoCo Menagerie and generally carry tuned actuator, armature and contact parameters, so
        they are usually the better choice for MuJoCo). Any other name is loaded as a URDF.

        NOTE: pinocchio cannot parse MJCF. When loading an `*_mj_description`, give
        `pinocchio_urdf_description` (e.g. `g1_description` alongside `g1_mj_description`) if the
        controllers used need a `PinocchioInterface`.

        Args:
            robot_description_name (str): The name of the robot description package. Should be a
                value available in the awesome robots list.
            ee_names (List[str], optional): List of end-effectors for the robot. Defaults to None.
            place_on_ground (bool): If true, the base position height will automatically be
                adjusted such that the robot is on the ground. Defaults to True.
            default_base_position (np.ndarray, optional): Default base position in the world frame.
                Defaults to None (np.zeros(3)).
            default_base_orientation (np.ndarray, optional): Default base orientation quaternion.
                Defaults to None (np.array([0, 0, 0, 1])).
            default_joint_positions (List[float], optional): Optional starting joint values.
                Defaults to None.
            create_pinocchio_interface (bool, optional): If true, creates a PinocchioInterface
                object for this robot. Defaults to True.
            pinocchio_urdf_description (str, optional): Description package name whose URDF should
                be used for the PinocchioInterface. Defaults to None.
            pinocchio_ee_names (List[str], optional): End-effector frame names as they appear in the
                URDF, when these differ from the MJCF body names. Defaults to None (use
                `ee_names`).
            floating_base (bool, optional): Whether this robot has a floating base. Defaults to
                True.
            render (bool, optional): Launch the MuJoCo viewer. Defaults to True.
            verbose (bool, optional): Verbosity flag. Defaults to True.

        Returns:
            MujocoRobot: A MujocoRobot instance for the specified robot description.
        """
        urdf_path = None
        mjcf_path = None
        if robot_description_name.endswith("_mj_description"):
            mjcf_path = get_mjcf_from_awesome_robot_descriptions(
                robot_description_pkg_name=robot_description_name
            )
        else:
            urdf_path = get_urdf_from_awesome_robot_descriptions(
                robot_description_pkg_name=robot_description_name
            )

        pinocchio_urdf_path = None
        if pinocchio_urdf_description is not None:
            pinocchio_urdf_path = get_urdf_from_awesome_robot_descriptions(
                robot_description_pkg_name=pinocchio_urdf_description
            )

        return cls(
            urdf_path=urdf_path,
            mjcf_path=mjcf_path,
            ee_names=ee_names,
            place_on_ground=place_on_ground,
            default_base_position=default_base_position,
            default_base_orientation=default_base_orientation,
            default_joint_positions=default_joint_positions,
            create_pinocchio_interface=create_pinocchio_interface,
            pinocchio_urdf_path=pinocchio_urdf_path,
            pinocchio_ee_names=pinocchio_ee_names,
            floating_base=floating_base,
            render=render,
            verbose=verbose,
            **sim_interface_kwargs,
        )
