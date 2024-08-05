"""Helper functions for updating provided GlobalMotionPlan objects for UI Interfaces."""

from typing import Callable, Dict, List, Any
from functools import partial
import numpy as np

from ....core.types import GlobalMotionPlan, PlannerMode
from ....core.logging import logging

# pylint: disable=C0116,C0103


class _GlobalPlanModifiers:
    """Helper functions for updating provided GlobalMotionPlan objects (internal use only)."""

    @staticmethod
    def set_linear_velocity(
        global_plan: GlobalMotionPlan,
        linear_velocity: np.ndarray,
        scaling: np.ndarray,
        prev_vel_scaling: np.ndarray,
    ) -> GlobalMotionPlan:
        assert len(linear_velocity) == 3
        global_plan.twist.linear = (
            prev_vel_scaling * global_plan.twist.linear + scaling * linear_velocity
        )
        return global_plan

    @staticmethod
    def set_rotational_velocity(
        global_plan: GlobalMotionPlan,
        rotational_velocity: np.ndarray,
        scaling: np.ndarray,
        prev_vel_scaling: np.ndarray,
    ) -> GlobalMotionPlan:
        assert len(rotational_velocity) == 3
        global_plan.twist.angular = (
            prev_vel_scaling * global_plan.twist.angular + scaling * rotational_velocity
        )
        return global_plan

    @staticmethod
    def set_plan_mode(global_plan: GlobalMotionPlan, planner_mode: PlannerMode):
        global_plan.planner_mode = planner_mode
        return global_plan

    @staticmethod
    def toggle_plan_mode(global_plan: GlobalMotionPlan, planner_modes: List[PlannerMode]):
        if global_plan.planner_mode != planner_modes[0]:
            global_plan.planner_mode = planner_modes[0]
        else:
            global_plan.planner_mode = planner_modes[1]
        return global_plan

    @staticmethod
    def increment_linear_velocity(
        global_plan: GlobalMotionPlan, linear_velocity: np.ndarray
    ) -> GlobalMotionPlan:
        assert len(linear_velocity) == 3
        global_plan.twist.linear += np.array(linear_velocity)
        return global_plan

    @staticmethod
    def increment_rotational_velocity(
        global_plan: GlobalMotionPlan, rotational_velocity: np.ndarray
    ) -> GlobalMotionPlan:
        assert len(rotational_velocity) == 3
        global_plan.twist.angular += np.array(rotational_velocity)
        return global_plan

    @staticmethod
    def increment_relative_position(global_plan: GlobalMotionPlan, position_delta: np.ndarray):
        assert len(position_delta) == 3
        global_plan.relative_pose.position += np.array(position_delta)
        return global_plan

    @staticmethod
    def increment_joint_position(
        global_plan: GlobalMotionPlan, joint_id: int, position_delta: float
    ):
        if global_plan.joint_references.joint_positions is not None:
            global_plan.joint_references.joint_positions[joint_id] += position_delta
        return global_plan


class GlobalPlanJointPositionIncrementer:
    """Helper to bind a button to change the value of different joints incrementally.
    One button can be used to increment the value of the joint by a specific number.
    Another button(s) can be used to change the index of the joint being modified.
    """

    def __init__(self, update_only_if_planner_activated: bool = True):
        self._check_and_update = update_only_if_planner_activated
        self.joint_id = 0

    def _check_ready(self, global_plan: GlobalMotionPlan):
        if self._check_and_update and global_plan.planner_mode != PlannerMode.CUSTOM:
            logging.warning(
                f"{self.__class__.__name__}: Planner is not activated. Not updating joint commands."
                " Set PlannerMode to PlannerMode.CUSTOM in the global plan."
            )
            return False
        return True

    def increment_joint_target(self, global_plan: GlobalMotionPlan, position_delta: float):
        """Use as binding for a button in the format:
        `{ <button_id>: partial(obj.increment_joint_target, position_delta=value) }`"""
        if not self._check_ready(global_plan=global_plan):
            return global_plan
        if global_plan.joint_references.joint_positions is not None:
            global_plan.joint_references.joint_positions[self.joint_id] += position_delta
        return global_plan

    def increase_joint_id(self, global_plan: GlobalMotionPlan):
        """
        Change the id of the joint being modified by incrementing the index by 1.
        Use the handle to this function directly as binding for a button in the format:
        `{ <button_id>: obj.increase_joint_id }`"""
        if not self._check_ready(global_plan=global_plan):
            return global_plan
        if global_plan.joint_references.joint_positions is None:
            return global_plan
        if self.joint_id >= len(global_plan.joint_references.joint_positions) - 1:
            self.joint_id = 0
        else:
            self.joint_id += 1
        return global_plan

    def decrease_joint_id(self, global_plan: GlobalMotionPlan):
        """
        Change the id of the joint being modified by reducing the index by 1.
        Use the handle to this function directly as binding for a button in the format:
        `{ <button_id>: obj.decrease_joint_id }`"""
        if not self._check_ready(global_plan=global_plan):
            return global_plan
        if global_plan.joint_references.joint_positions is None:
            return global_plan
        if self.joint_id <= 0:
            self.joint_id = len(global_plan.joint_references.joint_positions) - 1
        else:
            self.joint_id -= 1
        return global_plan


class Hold2SwapBindings:
    """Helper tool to make holding a button change the behaviour of another button/trigger."""

    def __init__(
        self,
        default_binding: Callable[[GlobalMotionPlan, Any], GlobalMotionPlan],
        alternate_binding: Callable[[GlobalMotionPlan, Any], GlobalMotionPlan],
    ):
        """Helper tool to make holding a button (called "B" here) change the behaviour of another
        button/trigger (called "button A" here).

        Args:
            default_binding (Callable[[GlobalMotionPlan, Any], GlobalMotionPlan]): The default
                behaviour of the binded button ("button A").
            alternate_binding (Callable[[GlobalMotionPlan, Any], GlobalMotionPlan]): Alternate
                behaviour of the binded button ("button A") when button "button B" is held.

        """
        self._default_binding = default_binding
        self._alternate_binding = alternate_binding
        self._trigger_held = False

    def trigger_swap(self, _any: Any):
        """Set this function as the binding for the trigger hold button ("button B")."""
        self._trigger_held = True
        return _any

    def use_binding(self, *args, **kwargs):
        """Set this function as the binding for the main button ("button A")."""
        if self._trigger_held:
            out = self._alternate_binding(*args, **kwargs)
        else:
            out = self._default_binding(*args, **kwargs)
        self._trigger_held = False
        return out


class Key2GlobalPlanHelper:
    """Helper class for creating keyboard mapppings for keyboard_interface.

    Provides static methods/objects that can be used for easily creating lambdas for creating
    key mappings (to be used for keyboard_interface)

    Each of the following will return a callable in the appropriate signature and can be used
    directly for the keyboard_interface. (see DEFAULT_KEYBOARD_MAPPING defined below)

    Available objects/methods (static):
        - CUSTOM
        - IDLE
        - HOLD_POSITION
        - APPLY_LIN_VEL(linear_velocity: np.ndarray)
        - APPLY_ROT_VEL(rotational_velocity: np.ndarray)
        - APPLY_LIN_VEL_DELTA(linear_velocity: np.ndarray)
        - APPLY_ROT_VEL_DELTA(rotational_velocity: np.ndarray)
    """

    CUSTOM: Callable[[GlobalMotionPlan], GlobalMotionPlan] = partial(
        _GlobalPlanModifiers.set_plan_mode, planner_mode=PlannerMode.CUSTOM
    )
    """static object that returns a callable. Sets the planner_mode to CUSTOM (controller
    activate)."""

    IDLE: Callable[[GlobalMotionPlan], GlobalMotionPlan] = partial(
        _GlobalPlanModifiers.set_plan_mode, planner_mode=PlannerMode.IDLE
    )
    """static object that returns a callable. Sets the planner_mode to IDLE."""

    HOLD_POSITION: Callable[[GlobalMotionPlan], GlobalMotionPlan] = partial(
        _GlobalPlanModifiers.set_plan_mode, planner_mode=PlannerMode.HOLD_POSITION
    )
    """static object that returns a callable. Sets the planner_mode to HOLD_POSITION."""

    TOGGLE_CUSTOM_IDLE: Callable[[GlobalMotionPlan], GlobalMotionPlan] = partial(
        _GlobalPlanModifiers.toggle_plan_mode,
        planner_modes=[PlannerMode.CUSTOM, PlannerMode.IDLE],
    )

    @staticmethod
    def APPLY_LIN_VEL(
        linear_velocity: np.ndarray,
    ) -> Callable[[GlobalMotionPlan], GlobalMotionPlan]:
        """Static method that returns a callable for setting linear velocity (takes one argument).

        Args:
            linear_velocity (np.ndarray): linear velocity (in base frame) to be set (m/sec)

        Returns:
            Callable[[GlobalMotionPlan], GlobalMotionPlan]: returns a callable that can be used by
                the keyboard_interface.
        """
        return partial(
            _GlobalPlanModifiers.set_linear_velocity,
            linear_velocity=linear_velocity,
            scaling=1,
            prev_vel_scaling=0,
        )

    @staticmethod
    def APPLY_ROT_VEL(
        rotational_velocity: np.ndarray,
    ) -> Callable[[GlobalMotionPlan], GlobalMotionPlan]:
        """Static method that returns a callable for setting rotational velocity (takes one
        argument).

        Args:
            rotational_velocity (np.ndarray): rotational velocity (in base frame) to be set
                (rad/sec)

        Returns:
            Callable[[GlobalMotionPlan], GlobalMotionPlan]: returns a callable that can be used by
                the keyboard_interface.
        """
        return partial(
            _GlobalPlanModifiers.set_rotational_velocity,
            rotational_velocity=rotational_velocity,
            scaling=1,
            prev_vel_scaling=0,
        )

    @staticmethod
    def APPLY_LIN_VEL_DELTA(
        linear_velocity: np.ndarray,
    ) -> Callable[[GlobalMotionPlan], GlobalMotionPlan]:
        """Static method that returns a callable for INCREMENTING current linear velocity (takes
            one argument).

        Args:
            linear_velocity (np.ndarray): linear velocity (in base frame) to be added (can be
                negative) (m/sec)

        Returns:
            Callable[[GlobalMotionPlan], GlobalMotionPlan]: returns a callable that can be used by
                the keyboard_interface.
        """
        return partial(
            _GlobalPlanModifiers.increment_linear_velocity,
            linear_velocity=linear_velocity,
        )

    @staticmethod
    def APPLY_ROT_VEL_DELTA(
        rotational_velocity: np.ndarray,
    ) -> Callable[[GlobalMotionPlan], GlobalMotionPlan]:
        """Static method that returns a callable for INCREMENTING current rotational velocity
            (takes one argument).

        Args:
            rotational_velocity (np.ndarray): rotational velocity (in base frame) to be added
                (can be negative) (rad/sec)

        Returns:
            Callable[[GlobalMotionPlan], GlobalMotionPlan]: returns a callable that can be used by
                the keyboard_interface.
        """
        return partial(
            _GlobalPlanModifiers.increment_rotational_velocity,
            rotational_velocity=rotational_velocity,
        )

    @staticmethod
    def APPLY_RELATIVE_POS_DELTA(
        position_delta: np.ndarray,
    ) -> Callable[[GlobalMotionPlan], GlobalMotionPlan]:
        """Static method that returns a callable for INCREMENTING position in teleop frame (takes
        one argument).

        Args:
            position_delta (np.ndarray): values to be added to current position of the robot base
                in the teleop frame

        Returns:
            Callable[[GlobalMotionPlan], GlobalMotionPlan]: returns a callable that can be used by
                the keyboard_interface.
        """
        return partial(
            _GlobalPlanModifiers.increment_relative_position,
            position_delta=position_delta,
        )

    @staticmethod
    def APPLY_JOINT_POS_DELTA(
        joint_id: int,
        position_delta: float,
    ) -> Callable[[GlobalMotionPlan], GlobalMotionPlan]:
        """Static method that returns a callable for INCREMENTING joint position for the specified
        joint id.

        Args:
            position_delta (float): value to be added to current joint position

        Returns:
            Callable[[GlobalMotionPlan], GlobalMotionPlan]: returns a callable that can be used by
                the keyboard_interface.
        """
        return partial(
            _GlobalPlanModifiers.increment_joint_position,
            joint_id=joint_id,
            position_delta=position_delta,
        )


class Analog2GlobalPlanHelper:
    """Helper class for creating mapppings for non-binary analog keys (e.g. analog gamepad stick).

    Provides static methods/objects that can be used for easily creating lambdas for creating
    key mappings (to be used for non-binary analog keys)

    Each of the following will return a callable in the appropriate signature and can be used
    directly for the non-binary analog keys. (see DEFAULT_KEYBOARD_MAPPING defined below)

    Available objects/methods (static):
        - CUSTOM
        - IDLE
        - HOLD_POSITION
        - APPLY_LIN_VEL(linear_velocity: np.ndarray)
        - APPLY_ROT_VEL(rotational_velocity: np.ndarray)
        - APPLY_LIN_VEL_DELTA(linear_velocity: np.ndarray)
        - APPLY_ROT_VEL_DELTA(rotational_velocity: np.ndarray)
    """

    @staticmethod
    def APPLY_LIN_VEL_SCALED(
        max_linear_velocity: np.ndarray,
        prev_vel_scaling: np.ndarray,
    ) -> Callable[[GlobalMotionPlan], GlobalMotionPlan]:
        """Static method that returns a callable for setting linear velocity (takes 2 arguments).

        Args:
            max_linear_velocity (np.ndarray): max linear velocity (in base frame) to be set (m/sec)
            prev_vel_scaling (np.ndarray): scaling to use on previous velocity. This is added to
                the new scaled linear velocity input.

        Returns:
            Callable[[GlobalMotionPlan, np.ndarray, np.ndarray], GlobalMotionPlan]: returns a
                callable that can be used by the joystick_interface with 2 arguments (global
                plan, analog np.ndarray scale value for each axis, np.ndarray scale value for
                previous value in the global plan message).
        """
        return lambda global_plan, analog_axis_scale: _GlobalPlanModifiers.set_linear_velocity(
            global_plan=global_plan,
            linear_velocity=max_linear_velocity,
            scaling=analog_axis_scale,
            prev_vel_scaling=prev_vel_scaling,
        )

    @staticmethod
    def APPLY_ROT_VEL_SCALED(
        max_rotational_velocity: np.ndarray,
        prev_vel_scaling: np.ndarray,
    ) -> Callable[[GlobalMotionPlan], GlobalMotionPlan]:
        """Static method that returns a callable for setting rotational velocity (takes 2
        arguments).

        Args:
            max_rotational_velocity (np.ndarray): max rotational velocity (in base frame) to be set
                (m/sec)
            prev_vel_scaling (np.ndarray): scaling to use on previous velocity. This is added to
                the new scaled rotational velocity input.

        Returns:
            Callable[[GlobalMotionPlan, np.ndarray, np.ndarray], GlobalMotionPlan]: returns a
                callable that can be used by the joystick_interface with 2 arguments (global plan,
                analog np.ndarray scale value for each axis, np.ndarray scale value for previous
                value in the global plan message).
        """
        return lambda global_plan, analog_axis_scale: _GlobalPlanModifiers.set_rotational_velocity(
            global_plan=global_plan,
            rotational_velocity=max_rotational_velocity,
            scaling=analog_axis_scale,
            prev_vel_scaling=prev_vel_scaling,
        )


def get_keymapping_doc(
    key_mappings: Dict[str, Callable[[GlobalMotionPlan], GlobalMotionPlan]],
):
    """Get a docstring for a given Mapping from string to callable."""
    out_string = "Key mappings:\n"
    docs = {}
    for k in key_mappings:
        if isinstance(key_mappings[k], partial):
            doc = f"{key_mappings[k].func.__name__}:\n\t\targs: {key_mappings[k].keywords}"
        else:
            doc = key_mappings[k].__name__
        docs[k] = doc
        out_string += f"\t{k}: {doc}\n"

    return out_string, docs
