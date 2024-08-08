"""A simple joystick interface following the GlobalMotionPlanner protocol.

This class allows using bindings to update a global motion plan from user joystick input.
The modified global motion plan object can then be retrieved when generate_global_plan
method is called in the control loop.
"""

from threading import Thread
import copy
from inputs import get_gamepad, UnpluggedError

from .ui_base import UIBase
from ....core.types import GlobalMotionPlan, RobotState
from .key_mappings import DEFAULT_GAMEPAD_MAPPINGS, GamePadMappings
from ....core.logging import logging
from ....core.exceptions import NotConnectedError


class JoystickGlobalPlannerInterface(UIBase):
    """A simple joystick interface following the GlobalMotionPlanner protocol."""

    def __init__(
        self,
        gamepad_mappings: GamePadMappings = DEFAULT_GAMEPAD_MAPPINGS,
        default_global_plan: GlobalMotionPlan = GlobalMotionPlan(),
        check_connection_at_init: bool = False,
    ):
        """A simple joystick interface following the GlobalMotionPlanner protocol.

        NOTE: This component initiates a thread for reading joystick input!

        This class allows using bindings to update a global motion plan from user joystick input.
        The modified global motion plan object can then be retrieved when generate_global_plan
        method is called in the control loop.

        Args:
            gamepad_mappings (GamePadMappings, optional): This object can be used to provide custom
                mappings from joystick input to updates for GlobalMotionPlan object (see
                key_mappings.py). Defaults to DEFAULT_GAMEPAD_MAPPINGS.
            default_global_plan (GlobalMotionPlan, optional): _the default to use when initialising
                (and when reset it called (to be implemented)). Defaults to GlobalMotionPlan().
            check_connection_at_init (bool, optional): If set to True, will check if joystick is
                connected during construction, and throw exception if not found. Defaults to False.

        Raises:
            NotConnectedError: If `check_connection_at_init` set to True and joystick could
                not be detected.
        """
        self._keep_alive = True
        self._runner_thread = Thread(target=self._read_thread)
        self._connection_verified = False
        if check_connection_at_init and not self._check_connection():
            self._keep_alive = False
            raise NotConnectedError("Could not find joystick.")

        self._global_plan = default_global_plan
        self._mappings = gamepad_mappings

    def _check_connection(self):
        logging.info(f"{self.__class__.__name__}: Waiting for joystick...")
        try:
            get_gamepad()
            logging.info(f"{self.__class__.__name__}: Joystick interface ready!")
            self._connection_verified = True
            return True
        except (IndexError, UnpluggedError):
            return False

    def _read_thread(self):
        if not self._connection_verified:
            self._check_connection()
        try:
            while self._keep_alive:
                events = get_gamepad()
                for event in events:
                    if event.ev_type == "Key" and event.state == 1:
                        if event.code == "BTN_START":
                            self._global_plan = self._mappings.button_mappings.START_BUTTON(
                                self._global_plan
                            )
                        elif event.code == "BTN_SOUTH":
                            self._global_plan = self._mappings.button_mappings.SOUTH(
                                self._global_plan
                            )
                        elif event.code == "BTN_NORTH":
                            self._global_plan = self._mappings.button_mappings.NORTH(
                                self._global_plan
                            )
                        elif event.code == "BTN_EAST":
                            self._global_plan = self._mappings.button_mappings.EAST(
                                self._global_plan
                            )
                        elif event.code == "BTN_WEST":
                            self._global_plan = self._mappings.button_mappings.WEST(
                                self._global_plan
                            )
                        elif event.code == "BTN_TR":
                            self._global_plan = self._mappings.button_mappings.RB(self._global_plan)
                        elif event.code == "BTN_TL":
                            self._global_plan = self._mappings.button_mappings.LB(self._global_plan)

                    elif event.ev_type == "Absolute":
                        if event.code == "ABS_Y":
                            self._global_plan = self._mappings.analog_mappings.LEFT_Y(
                                self._global_plan, -event.state / 32767.0
                            )
                        elif event.code == "ABS_X":
                            self._global_plan = self._mappings.analog_mappings.LEFT_X(
                                self._global_plan, -event.state / 32767.0
                            )
                        elif event.code == "ABS_RY":
                            self._global_plan = self._mappings.analog_mappings.RIGHT_Y(
                                self._global_plan, -event.state / 32767.0
                            )
                        elif event.code == "ABS_RX":
                            self._global_plan = self._mappings.analog_mappings.RIGHT_X(
                                self._global_plan, -event.state / 32767.0
                            )
                        elif event.code == "ABS_RZ":
                            self._global_plan = self._mappings.analog_mappings.RT(
                                self._global_plan, event.state / 255
                            )
                        elif event.code == "ABS_Z":
                            self._global_plan = self._mappings.analog_mappings.LT(
                                self._global_plan, event.state / 255
                            )
                        elif event.code == "ABS_HAT0Y":
                            if event.state == -1:
                                self._global_plan = self._mappings.button_mappings.DPAD_UP(
                                    self._global_plan
                                )
                            elif event.state == 1:
                                self._global_plan = self._mappings.button_mappings.DPAD_DOWN(
                                    self._global_plan
                                )
                        elif event.code == "ABS_HAT0X":
                            if event.state == -1:
                                self._global_plan = self._mappings.button_mappings.DPAD_LEFT(
                                    self._global_plan
                                )
                            elif event.state == 1:
                                self._global_plan = self._mappings.button_mappings.DPAD_RIGHT(
                                    self._global_plan
                                )
        except UnpluggedError:
            self._keep_alive = False

    def process_user_input(
        self, robot_state: RobotState = None, t: float = None, dt: float = None
    ) -> GlobalMotionPlan:
        if self._global_plan.joint_references.joint_names is None:
            self._global_plan.joint_references.joint_names = copy.deepcopy(
                robot_state.joint_states.joint_names
            )
            self._global_plan.joint_references.joint_positions = copy.deepcopy(
                robot_state.joint_states.joint_positions
            )
        if not self._keep_alive:
            self.shutdown()
            raise NotConnectedError("Could not read joystick input.")
        elif not self._runner_thread.is_alive():
            self._runner_thread.start()
        return copy.deepcopy(self._global_plan)

    def shutdown(self):
        super().shutdown()
        self._keep_alive = False
        if self._runner_thread.is_alive():
            self._runner_thread.join()
