from typing import Callable, Any, Tuple
from functools import partial
from dataclasses import dataclass, field
import numpy as np

from ....core.types import GlobalMotionPlan
from .ui_utils import (
    Key2GlobalPlanHelper,
    Analog2GlobalPlanHelper,
    GlobalPlanJointPositionIncrementer,
)

MAX_LIN_VEL = 0.5
MAX_ANG_VEL = 1.0
LIN_VEL_DELTA = 0.05
ANG_VEL_DELTA = 0.1
POS_DELTA = 0.02
JOINT_POS_DELTA = 0.01


_jpi = GlobalPlanJointPositionIncrementer()

DEFAULT_KEYBOARD_MAPPING = {
    "o": Key2GlobalPlanHelper.TOGGLE_CUSTOM_IDLE,
    "h": Key2GlobalPlanHelper.HOLD_POSITION,
    "w": Key2GlobalPlanHelper.APPLY_LIN_VEL_DELTA(np.array([LIN_VEL_DELTA, 0.0, 0.0])),
    "s": Key2GlobalPlanHelper.APPLY_LIN_VEL_DELTA(np.array([-LIN_VEL_DELTA, 0.0, 0.0])),
    "q": Key2GlobalPlanHelper.APPLY_LIN_VEL_DELTA(np.array([0.0, LIN_VEL_DELTA, 0.0])),
    "e": Key2GlobalPlanHelper.APPLY_LIN_VEL_DELTA(np.array([0.0, -LIN_VEL_DELTA, 0.0])),
    "a": Key2GlobalPlanHelper.APPLY_ROT_VEL_DELTA(np.array([0.0, 0.0, ANG_VEL_DELTA])),
    "d": Key2GlobalPlanHelper.APPLY_ROT_VEL_DELTA(np.array([0.0, 0.0, -ANG_VEL_DELTA])),
    "up": partial(_jpi.increment_joint_target, position_delta=JOINT_POS_DELTA),
    "down": partial(_jpi.increment_joint_target, position_delta=-JOINT_POS_DELTA),
    "right": _jpi.increase_joint_id,
    "left": _jpi.decrease_joint_id,
}
"""Default key mappings when using the keyboard interface for sending global motion
plan messages.

- o --> TOGGLE between CUSTOM and IDLE
- h --> HOLD_POSITION
- w --> APPLY_LIN_VEL_DELTA(np.array([0.05, 0.0, 0.0]))
- s --> APPLY_LIN_VEL_DELTA(np.array([-0.05, 0.0, 0.0]))
- q --> APPLY_LIN_VEL_DELTA(np.array([0.0, 0.05, 0.0]))
- e --> APPLY_LIN_VEL_DELTA(np.array([0.0, -0.05, 0.0]))
- a --> APPLY_ROT_VEL_DELTA(np.array([0.0, 0.0, 0.1]))
- d --> APPLY_ROT_VEL_DELTA(np.array([0.0, 0.0, -0.1]))
- up --> increment selected joint by 0.01 units
- down --> reduce reference joint position target for selected joint by 0.01 units
- right --> increment the id of the joint to be controlled with up/down keys
- left --> reduce the id of the joint to be controlled with up/down keys by 1
"""


def _return_arg1(x: Any, *args, **kwargs):  # pylint: disable=W0613
    return x


@dataclass
class GamePadButtonMappings:
    """By default all buttons do nothing to the GlobalMotionPlan object.

    Each button mapping can be replaced with another function callable
    that takes as argument a GlobalMotionPlan object and returns
    another GlobalMotionPlan object.
    """

    NORTH: Callable[[GlobalMotionPlan], GlobalMotionPlan] = _return_arg1
    SOUTH: Callable[[GlobalMotionPlan], GlobalMotionPlan] = _return_arg1
    EAST: Callable[[GlobalMotionPlan], GlobalMotionPlan] = _return_arg1
    WEST: Callable[[GlobalMotionPlan], GlobalMotionPlan] = _return_arg1
    DPAD_UP: Callable[[GlobalMotionPlan], GlobalMotionPlan] = _return_arg1
    DPAD_DOWN: Callable[[GlobalMotionPlan], GlobalMotionPlan] = _return_arg1
    DPAD_LEFT: Callable[[GlobalMotionPlan], GlobalMotionPlan] = _return_arg1
    DPAD_RIGHT: Callable[[GlobalMotionPlan], GlobalMotionPlan] = _return_arg1
    START_BUTTON: Callable[[GlobalMotionPlan], GlobalMotionPlan] = _return_arg1
    RB: Callable[[GlobalMotionPlan], GlobalMotionPlan] = _return_arg1
    LB: Callable[[GlobalMotionPlan], GlobalMotionPlan] = _return_arg1


@dataclass
class GamePadAnalogStickMappings:
    """By default all analog sticks do nothing to the GlobalMotionPlan object.

    Each analog stick mapping can be replaced with another function callable
    that takes as argument a GlobalMotionPlan object and a scaling value (float
    from the analog output (e.g. joystick value)), and returns a GlobalMotionPlan object.
    """

    LEFT_X: Callable[[GlobalMotionPlan, float], GlobalMotionPlan] = _return_arg1
    LEFT_Y: Callable[[GlobalMotionPlan, float], GlobalMotionPlan] = _return_arg1
    RIGHT_X: Callable[[GlobalMotionPlan, float], GlobalMotionPlan] = _return_arg1
    RIGHT_Y: Callable[[GlobalMotionPlan, float], GlobalMotionPlan] = _return_arg1
    RT: Callable[[GlobalMotionPlan, float], GlobalMotionPlan] = _return_arg1
    LT: Callable[[GlobalMotionPlan, float], GlobalMotionPlan] = _return_arg1


@dataclass
class GamePadMappings:
    button_mappings: GamePadButtonMappings = field(default_factory=GamePadButtonMappings)
    analog_mappings: GamePadAnalogStickMappings = field(default_factory=GamePadAnalogStickMappings)


def toggle_mapping(
    mapping: GamePadButtonMappings | GamePadAnalogStickMappings,
    key_name: str,
    bindings: Tuple[Callable[..., GlobalMotionPlan], Callable[..., GlobalMotionPlan]],
):
    """Toggle between two bindings (callables) for the specified attribute name in the mapping.

    Args:
        mapping (GamePadButtonMappings | GamePadAnalogStickMappings): The original mapping (should
            be mutable).
        key_name (str): Name of the attribute whose binding is to be modified.
        bindings (Tuple[Callable[..., GlobalMotionPlan], Callable[..., GlobalMotionPlan]]): The two
            bindings to switch between.
    """
    if getattr(mapping, key_name) == bindings[0]:
        setattr(mapping, key_name, bindings[1])
    else:
        setattr(mapping, key_name, bindings[0])


def _toggle_mapping_and_return_original_gplan(
    global_plan: GlobalMotionPlan,
    mapping: GamePadButtonMappings | GamePadAnalogStickMappings,
    key_name: str,
    bindings: Tuple[Callable[..., GlobalMotionPlan], Callable[..., GlobalMotionPlan]],
):
    toggle_mapping(mapping=mapping, key_name=key_name, bindings=bindings)
    return global_plan


_jpi2 = GlobalPlanJointPositionIncrementer()

DEFAULT_GAMEPAD_MAPPINGS = GamePadMappings(
    button_mappings=GamePadButtonMappings(
        START_BUTTON=Key2GlobalPlanHelper.TOGGLE_CUSTOM_IDLE,
        SOUTH=Key2GlobalPlanHelper.HOLD_POSITION,
        # DPAD_UP=Key2GlobalPlanHelper.APPLY_RELATIVE_POS_DELTA(
        #     position_delta=[0.0, 0.0, POS_DELTA]
        # ), # change base pose target height
        # DPAD_DOWN=Key2GlobalPlanHelper.APPLY_RELATIVE_POS_DELTA(
        #     position_delta=[0.0, 0.0, -POS_DELTA]
        # ), # change base pose target height
        DPAD_UP=partial(_jpi.increment_joint_target, position_delta=JOINT_POS_DELTA),
        DPAD_DOWN=partial(_jpi.increment_joint_target, position_delta=-JOINT_POS_DELTA),
        DPAD_LEFT=_jpi.decrease_joint_id,
        DPAD_RIGHT=_jpi.increase_joint_id,
    ),
    analog_mappings=GamePadAnalogStickMappings(
        LEFT_X=Analog2GlobalPlanHelper.APPLY_LIN_VEL_SCALED(
            max_linear_velocity=np.array([0.0, MAX_LIN_VEL, 0.0]),
            prev_vel_scaling=np.array([1.0, 0.0, 1.0]),
        ),
        LEFT_Y=Analog2GlobalPlanHelper.APPLY_LIN_VEL_SCALED(
            max_linear_velocity=np.array([MAX_LIN_VEL, 0.0, 0.0]),
            prev_vel_scaling=np.array([0.0, 1.0, 1.0]),
        ),
        RIGHT_X=Analog2GlobalPlanHelper.APPLY_ROT_VEL_SCALED(
            max_rotational_velocity=np.array([0.0, 0.0, MAX_ANG_VEL]),
            prev_vel_scaling=np.array([1.0, 1.0, 0.0]),
        ),
        RIGHT_Y=Analog2GlobalPlanHelper.APPLY_ROT_VEL_SCALED(
            max_rotational_velocity=np.array([0.0, MAX_ANG_VEL, 0.0]),
            prev_vel_scaling=np.array([1.0, 0.0, 1.0]),
        ),
    ),
)
"""Default joystick mapping when using JoystickInterface as global planner for sending
global plan messages.

NOTE: SOUTH, EAST, WEST, NORTH below indicates the keys on the right side of the joypad.
    (e.g. this is A, B, X, Y respectively on logitech joypad)

BUG: WEST and NORTH for logitech seems to be switched!!

- Start btn --> Toggle between CUSTOM and IDLE modes
- South --> HOLD_POSITION
- DPAD UP --> Increase joint position target for the current joint id by 0.01 units.
- DPAD DOWN --> Decrease joint position target for the current joint id by 0.01 units.
- DPAD LEFT --> Change the ID of the joint by -1 (affects actions of DPAD UP and DOWN buttons).
- DPAD LEFT --> Change the ID of the joint by +1 (affects actions of DPAD UP and DOWN buttons).
- LEFT Analog up/down --> apply linear velocity forward/backward (x axis) (max vel: 0.5 m/s)
    (OR) apply linear velocity up/down (z axis) (max vel: 0.5 m/s). Toggle behaviour of
    LEFT Analog up/down using `LB` button.
- LEFT Analog right/left --> apply linear velocity right/left (base frame) (max vel: 0.5 m/s)
- RIGHT Analog right/left --> apply rotational velocity (about z axis of base frame; yaw rate)
    (max vel: 1.0 rad/sec) (OR) apply rotational velocity (about x axis; roll rate)
    (max vel: 0.5 m/s). Toggle behaviour of RIGHT Analog right/left using `RB` button.
- RIGHT Analog up/down --> apply rotational velocity (about y axis of global frame; pitch rate)

- LB: This button changes the behaviour of the LEFT analog stick. Switches between sending
    linear velocity along X, and linear velocity along Z.
- RB: This button changes the behaviour of the RIGHT analog stick. Switches between sending
    angular velocity about X, and linear velocity about Z.
"""

# NOTE: Very ugly way of toggling what the analogs sticks do when RB, LB is pressed
DEFAULT_GAMEPAD_MAPPINGS.button_mappings.LB = partial(
    _toggle_mapping_and_return_original_gplan,
    mapping=DEFAULT_GAMEPAD_MAPPINGS.analog_mappings,
    key_name="LEFT_Y",
    bindings=[
        Analog2GlobalPlanHelper.APPLY_LIN_VEL_SCALED(
            max_linear_velocity=np.array([0.0, 0.0, MAX_LIN_VEL]),
            prev_vel_scaling=np.array([1.0, 1.0, 0.0]),
        ),
        Analog2GlobalPlanHelper.APPLY_LIN_VEL_SCALED(
            max_linear_velocity=np.array([MAX_LIN_VEL, 0.0, 0.0]),
            prev_vel_scaling=np.array([0.0, 1.0, 1.0]),
        ),
    ],
)
DEFAULT_GAMEPAD_MAPPINGS.button_mappings.RB = partial(
    _toggle_mapping_and_return_original_gplan,
    mapping=DEFAULT_GAMEPAD_MAPPINGS.analog_mappings,
    key_name="RIGHT_X",
    bindings=[
        Analog2GlobalPlanHelper.APPLY_ROT_VEL_SCALED(
            max_rotational_velocity=np.array([-MAX_ANG_VEL, 0.0, 0.0]),
            prev_vel_scaling=np.array([0.0, 1.0, 1.0]),
        ),
        Analog2GlobalPlanHelper.APPLY_ROT_VEL_SCALED(
            max_rotational_velocity=np.array([0.0, 0.0, MAX_ANG_VEL]),
            prev_vel_scaling=np.array([1.0, 1.0, 0.0]),
        ),
    ],
)
