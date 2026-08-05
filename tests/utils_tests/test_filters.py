import numpy as np
import pytest

from pyrcf.utils.filters import (
    abs_bounded_derivative_filter,
    bounded_derivative_filter,
    first_order_filter,
    low_pass_filter,
    second_order_filter,
)


class TestFirstOrderFilter:

    def test_zero_gain_holds_current_value(self):
        assert first_order_filter(1.0, 5.0, 0.0) == 1.0

    def test_unit_gain_jumps_to_desired(self):
        assert first_order_filter(1.0, 5.0, 1.0) == 5.0

    def test_interpolates(self):
        assert first_order_filter(0.0, 10.0, 0.25) == 2.5

    def test_works_elementwise_on_arrays(self):
        out = first_order_filter(np.zeros(3), np.array([1.0, 2.0, 4.0]), 0.5)
        assert np.allclose(out, np.array([0.5, 1.0, 2.0]))

    def test_gain_above_one_rejected(self):
        with pytest.raises(AssertionError):
            first_order_filter(0.0, 1.0, 1.5)


class TestSecondOrderFilter:

    def test_zero_gain_holds_current_value(self):
        assert second_order_filter(3.0, 9.0, 0.0) == 3.0

    def test_converges_towards_desired(self):
        value = 0.0
        for _ in range(5000):
            value = second_order_filter(value, 1.0, 0.05)
        assert value == pytest.approx(1.0, abs=1e-3)

    def test_stays_between_current_and_desired(self):
        out = second_order_filter(0.0, 1.0, 0.3)
        assert 0.0 < out < 1.0

    def test_effective_gain_is_squared(self):
        """A single step moves by `gain**2` of the error, not `gain` -- this is what makes it
        a slower/smoother interpolation than `first_order_filter` for the same gain."""
        gain = 0.05
        assert second_order_filter(0.0, 1.0, gain) == pytest.approx(gain**2)
        assert second_order_filter(0.0, 1.0, gain) < first_order_filter(0.0, 1.0, gain)


class TestBoundedDerivativeFilter:

    def test_clamps_derivative(self):
        # asking for a jump of 10.0 in 0.1s = rate of 100, clamped to 1.0
        out = bounded_derivative_filter(0.0, 10.0, 0.1, (-100.0, 100.0), (-1.0, 1.0))
        assert out == pytest.approx(0.1)

    def test_clamps_output(self):
        out = bounded_derivative_filter(0.0, 10.0, 1.0, (-0.5, 0.5), (-100.0, 100.0))
        assert out == pytest.approx(0.5)

    def test_abs_variant_matches_symmetric_bounds(self):
        assert abs_bounded_derivative_filter(0.0, 10.0, 0.1, 100.0, 1.0) == pytest.approx(
            bounded_derivative_filter(0.0, 10.0, 0.1, (-100.0, 100.0), (-1.0, 1.0))
        )


class TestLowPassFilter:

    def test_decays_towards_new_input(self):
        """Argument order is (prev_output, new_input, dt, cutoff_period)."""
        out = low_pass_filter(prev_output=1.0, new_input=0.0, dt=0.005, cutoff_period=1.0)
        assert out == pytest.approx(0.995)
        assert out < 1.0, "filter must move towards the new input"

    def test_converges_to_new_input(self):
        value = 1.0
        for _ in range(5000):
            value = low_pass_filter(value, 0.0, dt=0.005, cutoff_period=1.0)
        assert value == pytest.approx(0.0, abs=1e-6)

    def test_missing_timing_info_uses_default_alpha(self):
        assert low_pass_filter(0.0, 1.0) == pytest.approx(0.499)
        assert low_pass_filter(0.0, 1.0, dt=0.01) == pytest.approx(0.499)

    def test_zero_cutoff_period_disables_filtering(self):
        assert low_pass_filter(5.0, 1.0, dt=0.01, cutoff_period=0.0) == 1.0

    def test_aliasing_is_rejected(self):
        with pytest.raises(AssertionError, match="Nyquist|argument order|must be < 0.5"):
            low_pass_filter(0.0, 1.0, dt=1.0, cutoff_period=0.1)
