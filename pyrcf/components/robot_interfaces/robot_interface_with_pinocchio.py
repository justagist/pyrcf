from typing import List

from ...core.types import RobotState, Pose3D
from .robot_interface_base import RobotInterface
from ...utils.kinematics_dynamics import PinocchioInterface
from ...core.logging import throttled_logging


class RobotInterfaceWithPinocchio(RobotInterface):
    """
    Extension to the RobotInterface base class to provide easier interfacing with
    PinocchioInterface.

    NOTE: Any class inheriting this base class should call the constructor using
    ```
    super().__init__(
        robot_urdf=robot_urdf,
        ee_names=ee_names,
        floating_base=floating_base,
        verbose=verbose,
        create_pinocchio_interface=create_pinocchio_interface,
    )
    ```

    To update pinocchio in the control loop at every time, call
    `self.update_pinocchio_robot_state(robot_state=robot_state)` in the read() method
    implementation of the child class
    """

    def __init__(
        self,
        robot_urdf: str = None,
        ee_names: List[str] = (),
        floating_base: bool = False,
        verbose: bool = True,
        create_pinocchio_interface: bool = True,
    ):
        """Extension to the RobotInterface base class to provide easier interfacing with
        PinocchioInterface.

        NOTE: Any class inheriting this base class should call the constructor using
        ```
        super().__init__(
            robot_urdf=robot_urdf,
            ee_names=ee_names,
            floating_base=floating_base,
            verbose=verbose,
            create_pinocchio_interface=create_pinocchio_interface,
        )
        ```

        To update pinocchio in the control loop at every time, call
        `self.update_pinocchio_robot_state(robot_state=robot_state)` in the read() method
        implementation of the child class.


        Args:
            robot_urdf (str): Path to urdf file of robot.
            ee_names (List[str], optional): List of end-effectors for the robot. Defaults to None.
            floating_base (bool, optional): Specifying whether this robot has a floating or fixed
                base (setting to True creates a floating base instance with pinocchio, and makes
                the simulated robot free floating (not fixed base)). Defaults to True.
            verbose (bool, optional): Verbosity flag for debugging robot info during construction.
                Defaults to True.
            create_pinocchio_interface (bool, optional): If true, creates a PinocchioInterface
                object (mutable) for this robot.
        """
        self._pin_interface: PinocchioInterface = None
        self._urdf_path = robot_urdf
        if create_pinocchio_interface and robot_urdf is not None:
            if floating_base is None:
                raise ValueError(
                    f"{self.__class__.__name__}: Argument `floating_base` should be provided "
                    "to create PinocchioInterface."
                )
            self._pin_interface = PinocchioInterface(
                urdf_filename=robot_urdf,
                floating_base=floating_base,
                verbose=verbose,
                ee_names=ee_names,
            )

    def get_pinocchio_interface(self) -> PinocchioInterface:
        """Get the pinocchio interface associated with this robot.

        NOTE: this is a mutable object, and will be updated automatically
            whenever the read() method of this class instance is
            called.

        Returns:
            PinocchioInterface: A mutable copy of the PinocchioInterface object for
                this robot.
        """
        if self._pin_interface is None:
            raise ValueError(
                f"{self.__class__.__name__}: This robot was not initialised with a pinocchio interface! Set "
                "`robot_urdf=<path/to/combined_robot_urdf>` and `create_pinocchio_interface=True`"
                " during construction of object."
            )
        return self._pin_interface

    def update_pinocchio_robot_state(
        self, robot_state: RobotState, update_ee_states: bool = True
    ) -> RobotState:
        """Update the PinocchioInterface kinematics and dynamics data using the provided
        robot state, and return update robot state with info from pinocchio.

        Args:
            robot_state (RobotState): RobotState read from the robot.
            update_ee_states (bool, optional): If True, will populate
            `robot_state.state_estimates.end_effector_states` data using data from pinocchio.
            Defaults to True.

        Returns:
            RobotState: updated RobotState object.
        """
        if self._pin_interface is not None:
            self._pin_interface.update(
                global_base_position=robot_state.state_estimates.pose.position,
                global_base_quaternion=robot_state.state_estimates.pose.orientation,
                local_base_velocity_linear=robot_state.state_estimates.twist.linear,
                local_base_velocity_angular=robot_state.state_estimates.twist.angular,
                joint_positions=robot_state.joint_states.joint_positions,
                joint_velocities=robot_state.joint_states.joint_velocities,
                joint_order=robot_state.joint_states.joint_names,
            )
            if update_ee_states:
                robot_state.state_estimates.end_effector_states.ee_poses = [
                    Pose3D(*ee_pose) for ee_pose in self._pin_interface.get_ee_poses()
                ]
                robot_state.state_estimates.end_effector_states.ee_names = (
                    self._pin_interface.ee_names
                )
        else:
            throttled_logging.warning(
                f"{self.__class__.__name__}: Pinocchio interface does not "
                "exist for this robot. Not updating pinocchio or robot state.",
                delay_sec=2,
            )
        return robot_state

    @property
    def has_pinocchio_interface(self) -> bool:
        """Check if this robot has a usable pinocchio interface.

        Returns:
            bool: True if the robot interface has a PinocchioInterface.
        """
        return self._pin_interface is not None

    @property
    def urdf_path(self) -> str:
        """Path to the urdf used for creating this robot's pinocchio model."""
        return self._urdf_path
