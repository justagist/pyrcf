from ..components.robot_interfaces.robot_interface_base import RobotInterface
from ..components.global_planners.ui_reference_generators import (
    KeyboardGlobalPlannerInterface,
    JoystickGlobalPlannerInterface,
)
from ..components.controllers.controller_base import ControllerBase
from ..components.local_planners import LocalPlannerBase, BlindForwardingPlanner
from ..components.global_planners.global_planner_base import GlobalMotionPlannerBase
from ..components.state_estimators.state_estimator_base import (
    StateEstimatorBase,
    DummyStateEstimator,
)
from .simple_managed_ctrl_loop import SimpleManagedCtrlLoop
from ..components.agents.planner_controller_agent import PlannerControllerAgent
from ..components.controller_manager.simple_controller_manager import SimpleControllerManager
from ..core.logging import logging
from ..core.exceptions import NotConnectedError


class MinimalCtrlLoop(SimpleManagedCtrlLoop):
    """A minimal control loop with only 1 controller and 1 local planner.

    The run method runs the following in a loop of specified frequency.

    1. read robot state from robot interface
    2. update robot state with state estimates from state estimator
    3. get global plan message from global planner / user interface
    4. generate local plan using the global plan message
    5. compute control command to track the local plan
    6. write control command to robot
    """

    def __init__(
        self,
        robot_interface: RobotInterface,
        state_estimator: StateEstimatorBase,
        controller: ControllerBase,
        local_planner: LocalPlannerBase,
        global_planner: GlobalMotionPlannerBase,
        verbose: bool = True,
    ):
        """Create a minimal control loop where all components are run in sequence.

        Args:
            robot_interface (RobotInterface): The robot interface to use in the loop.
            state_estimator (StateEstimatorBase): State estimator object to use in the loop.
            controller (ControllerBase): The controller to be used.
            local_planner (LocalPlannerBase): The local planner object to be used in the loop.
            global_planner (GlobalMotionPlannerBase): The global planner object to be used in the
                loop.
        """
        controller_manager = SimpleControllerManager().add_agent(
            agent=PlannerControllerAgent(local_planner=local_planner, controller=controller)
        )
        super().__init__(
            robot_interface=robot_interface,
            state_estimator=state_estimator,
            controller_manager=controller_manager,
            global_planner=global_planner,
            verbose=verbose,
        )

    @classmethod
    def useWithDefaults(
        cls: "MinimalCtrlLoop",
        robot_interface: RobotInterface,
        controller: ControllerBase,
        global_planner: GlobalMotionPlannerBase = None,
        local_planner: LocalPlannerBase = BlindForwardingPlanner(),
        state_estimator: StateEstimatorBase = DummyStateEstimator(squawk=False),
        verbose: bool = True,
    ) -> "MinimalCtrlLoop":
        """Create a control loop default values for planners and state estimators.

        Default local planner: BlindForwardingPlanner
        Default global planner: Try JoystickGlobalPlannerInterface; fallback KeyboardGlobalPlannerInterface
        Default state estimator: DummyStateEstimator

        Args:
            robot_interface (RobotInterface): The robot interface to use in the loop.
            controller (ControllerBase): The controller to be used.
            global_planner (GlobalMotionPlannerBase, optional): The global planner to use. If None
                provided, will try to first find a joystick, and then creates a keyboard interface
                if joystick is not found.
            local_planner (LocalPlannerBase, optional): Local planner to use. Defaults to
                BlindForwardingPlanner.
            state_estimator (StateEstimatorBase, optional): State estimator to use. Defaults to
                DummyStateEstimator(squawk=False).

        Returns:
            MinimalCtrlLoop: returns a valid control loop object with planners and state estimators
                automatically defined.
        """

        if global_planner is None:
            try:
                global_planner = JoystickGlobalPlannerInterface(check_connection_at_init=True)
                logging.info("Using Joystick interface.")
            except (IndexError, NotConnectedError):
                logging.info("Could not detect joystick. Using Keyboard interface.")
                global_planner = KeyboardGlobalPlannerInterface()

        return cls(
            robot_interface=robot_interface,
            state_estimator=state_estimator,
            controller=controller,
            local_planner=local_planner,
            global_planner=global_planner,
            verbose=verbose,
        )
