from typing import Tuple, TypeAlias
import numpy as np
import quaternion

QuatType: TypeAlias = np.ndarray
"""Numpy array representating quaternion in format [x,y,z,w]"""


def bounded_derivative_filter(
    prev_output: float,
    new_input: float,
    dt: float,
    output_bounds: Tuple[float, float],
    derivative_bounds: Tuple[float, float],
) -> float:
    """
    Filter signal so that its output and output derivative stay within bounds.

    Args:
        prev_output (float): Previous filter output, or initial value.
        new_input (float): New filter input.
        dt (float): Sampling period in [s].
        output_bounds (Tuple[float, float]): Min and max value for the output.
        derivative_bounds (Tuple[float, float]): Min and max value for the output derivative.

    Returns:
        float: New filter output.
    """
    derivative = (new_input - prev_output) / dt
    derivative = np.clip(derivative, *derivative_bounds)
    output = prev_output + derivative * dt
    return np.clip(output, *output_bounds)


def abs_bounded_derivative_filter(
    prev_output: float,
    new_input: float,
    dt: float,
    max_output: float,
    max_derivative: float,
) -> float:
    """
    Filter signal so that the absolute values of its output and output
    derivative stay within bounds.

    Args:
        prev_output (float): Previous filter output, or initial value.
        new_input (float): New filter input.
        dt (float): Sampling period in [s].
        max_output (float): Maximum absolute value of the output.
        max_derivative (float): Maximum absolute value of the output derivative.

    Returns:
        float: New filter output.
    """
    return bounded_derivative_filter(
        prev_output,
        new_input,
        dt,
        (-max_output, max_output),
        (-max_derivative, max_derivative),
    )


def first_order_filter(
    current_value: float | np.ndarray,
    desired_value: float | np.ndarray,
    gain: float,
) -> float | np.ndarray:
    """First order filter.

    Args:
        current_value (float | np.ndarray): Current value.
        desired_value (float | np.ndarray): Target value.
        gain (float): Filter gain (applied to target during interpolation).

    Returns:
        float | np.ndarray: Output from filter.
    """
    assert gain <= 1.0
    return current_value + gain * (desired_value - current_value)


def second_order_filter(
    current_value: float | np.ndarray,
    desired_value: float | np.ndarray,
    gain: float,
) -> float | np.ndarray:
    """Second-order filter for data stream, interpolating from current to desired value.

    Args:
        current_value (float | np.ndarray): Current value.
        desired_value (float | np.ndarray): Target value.
        gain (float): Filter gain (applied to first-order filtered target during interpolation)

    Returns:
        float | np.ndarray: Output from filter.
    """
    return (1 - gain) * current_value + gain * first_order_filter(
        current_value=current_value, desired_value=desired_value, gain=gain
    )


def low_pass_filter(
    prev_output: float | np.ndarray,
    new_input: float | np.ndarray,
    dt: float = None,
    cutoff_period: float = None,
) -> float | np.ndarray:
    """
    Low-pass filter.

    Args:
        prev_output (float|np.ndarray): Previous filter output, or initial value.
        new_input (float|np.ndarray): New filter input.
        dt (float): Sampling period in [s]. Defaults to
            None (will select based on Nyquist-Shannon limit to avoid aliasing).
        cutoff_period (float, optional): Time constant of the filter in [s]. Defaults to
            None (will select based on Nyquist-Shannon limit to avoid aliasing).

        NOTE: dt/cutoff_period should be < 0.5 to respect Nyquist-Shannon sampling theorem.

    Returns:
        float|np.ndarray: New filter output.
    """
    if cutoff_period is None or dt is None:
        alpha = 0.499
    else:
        if cutoff_period is None or cutoff_period == 0.0:
            cutoff_period = 1e-6
        alpha = dt / cutoff_period
        assert alpha < 0.5  # Nyquist-Shannon sampling theorem
    return first_order_filter(current_value=prev_output, desired_value=new_input, gain=alpha)


def quaternion_slerp(quat1: QuatType, quat2: QuatType, interpolation_param: float) -> QuatType:
    """Get the slerp smoothed quaternion value between quat1 and quat2 at the time
    ratio given by `interpolation_param`.

    Args:
        quat1 (QuatType): Starting quaternion
        quat2 (QuatType): Target quaternion
        interpolation_param (float): Value between 0 and 1. 0 = quat1, 1 = quat2.

    Returns:
        QuatType: Numpy array representing a quaternion in order (x,y,z,w)
    """
    q = quaternion.slerp_evaluate(
        quaternion.quaternion(quat1[3], quat1[0], quat1[1], quat1[2]),
        quaternion.quaternion(quat2[3], quat2[0], quat2[1], quat2[2]),
        interpolation_param,
    )
    return np.array([q.x, q.y, q.z, q.w])
