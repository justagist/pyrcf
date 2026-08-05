"""A simple keyboard interface following the GlobalMotionPlanner protocol."""

from typing import Callable, Dict, Tuple
import copy
import pygame

from .ui_base import UIBase
from ....core.exceptions import CtrlLoopExitSignal
from ....core.logging import logger
from ....core.types import GlobalMotionPlan, RobotState
from .key_mappings import DEFAULT_KEYBOARD_MAPPING
from .ui_utils import get_keymapping_doc


def blit_multiline_text(surface, text, pos, font, line_spacing=3, color=None):
    """Function to display multiline text in pygame screen."""
    if color is None:
        color = pygame.Color("white")
    # 2D array where each row is a list of words.
    words = [word.split(" ") for word in text.splitlines()]
    x, y = pos
    for line in words:
        for word in line:
            tab_words = word.split("\t")
            for twd in tab_words:
                if twd == "":
                    twrd = "    "
                else:
                    twrd = twd
                word_surface = font.render(twrd, 0, color)
                word_width, word_height = word_surface.get_size()
                if x + word_width >= surface.get_size()[0]:
                    x = pos[0]  # Reset the x
                    y += word_height  # Start on new row
                surface.blit(word_surface, (x, y))
                x += word_width + font.size(" ")[0]
        x = pos[0]  # Reset the x
        y += word_height + line_spacing  # Start on new row


class KeyboardGlobalPlannerInterface(UIBase):
    """A simple keyboard interface inheriting from the GlobalMotionPlanner format."""

    def __init__(
        self,
        key_mappings: Dict[str, Callable[[GlobalMotionPlan], GlobalMotionPlan]] = None,
        default_global_plan: GlobalMotionPlan = None,
        window_size: Tuple[int, int] = (1000, 1000),
        parallel_mode: bool = False,
        verbose: bool = False,
    ) -> None:
        """A simple keyboard interface inheriting from the GlobalMotionPlanner format.

        This class allows using keybindings to update a global motion plan from user keyboard input.
        The modified global motion plan object can then be retrieved when generate_global_plan
        method is called in the control loop.

        NOTE: The pygame window should be in focus for the inputs to be recorded by this class.
        NOTE: Closing the pygame window requests a clean shutdown of the control loop.

        Args:
            key_mappings (dict[ str, callable[[GlobalMotionPlan], GlobalMotionPlan] ], optional):
                A char --> callable dictionary mapping a key from the keyboard to a function that
                updates a GlobalMotionPlan object. Defaults to DEFAULT_KEYBOARD_MAPPING.
            default_global_plan (GlobalMotionPlan, optional): the default to use when initialising
                (and when reset it called (to be implemented)). Defaults to GlobalMotionPlan().
            window_size (Tuple[int, int]): pygame window size in pixels. Defaults to (500, 500).
            parallel_mode (bool): DEPRECATED and ignored. Keyboard input is always polled
                synchronously from `process_user_input` (i.e. once per control loop iteration,
                which is far more often than a user can type). This option previously started a
                thread that drained the pygame event queue exactly once and then exited, so no
                input was ever recorded after construction. It cannot be fixed by looping either:
                pygame's event queue may only be pumped from the thread that created the display.
            verbose (bool): If true, will print command on console every time user inputs a valid
                key.
        """
        self._global_plan = (
            GlobalMotionPlan() if default_global_plan is None else default_global_plan
        )
        self._key_mappings = key_mappings if key_mappings is not None else DEFAULT_KEYBOARD_MAPPING
        self._verbose = verbose
        if parallel_mode:
            logger.warning(
                f"{self.__class__.__name__}: `parallel_mode` is deprecated and ignored. Keyboard"
                " input is polled synchronously in the control loop instead (pygame's event queue"
                " can only be pumped from the thread that created the display)."
            )

        pygame.init()
        pygame.mixer.quit()  # disable audio (otherwise alsa overrun warnings)

        self._display = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Keyboard input -- mock global planner")

        self._font = pygame.font.SysFont("Ubuntu", 12)

        self._doc_text, self._key_docs = get_keymapping_doc(self._key_mappings)

        blit_multiline_text(self._display, self._doc_text, (10, 10), self._font)

        pygame.display.update()

    def _update_global_plan_from_user_input(self) -> None:
        """Drain the pygame event queue and apply any mapped key presses to the global plan.

        Raises:
            CtrlLoopExitSignal: If the user closed the pygame window.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # let the control loop shut every component down cleanly instead of tearing
                # pygame down underneath ourselves and failing on the next poll
                logger.info(f"{self.__class__.__name__}: Window closed; requesting loop shutdown.")
                raise CtrlLoopExitSignal

            if event.type == pygame.KEYDOWN:
                c = pygame.key.name(event.key)
                if c in self._key_mappings:
                    self._global_plan = self._key_mappings[c](self._global_plan)
                    msg = f"{self.__class__.__name__}: User input received: Key {c}\n"
                    msg += f"{self._key_docs[c]}\n\nNew global plan: {self._global_plan}"
                    if self._verbose:
                        print(msg)
                    self._display.fill((0, 0, 0))
                    blit_multiline_text(
                        self._display,
                        self._doc_text + "\n\n" + msg,
                        (10, 10),
                        self._font,
                    )
                    pygame.display.update()

    def process_user_input(
        self, robot_state: RobotState, t: float = None, dt: float = None
    ) -> GlobalMotionPlan:
        if self._global_plan.joint_references.joint_names is None:
            self._global_plan.joint_references.joint_names = copy.deepcopy(
                robot_state.joint_states.joint_names
            )
            self._global_plan.joint_references.joint_positions = copy.deepcopy(
                robot_state.joint_states.joint_positions
            )
        self._update_global_plan_from_user_input()
        # return a copy so that downstream components cannot mutate this interface's own plan
        # (matches JoystickGlobalPlannerInterface)
        return copy.deepcopy(self._global_plan)

    def shutdown(self):
        super().shutdown()
        try:
            pygame.quit()
        except TypeError:
            pass
