"""
Tick-spacing tests for the bubble size distribution grids.

``nice_tick_step`` picks the x-axis major tick spacing for the histogram grids in
``grid_xanthan_by_placement``. The contract it must satisfy: never place more than
``max_ticks`` labels across the axis span, and only use spacings that read cleanly
(1, 2, 2.5 or 5 times a power of ten).
"""

import math

import pytest

from klarity.viz import nice_tick_step


@pytest.mark.parametrize(
    "span, max_ticks, expected",
    [
        # 10 mm: a 2.0 step admits 6 ticks (0, 2, ..., 10), so 2.5 wins with 5.
        (10.0, 5, 2.5),
        # 8 mm: a 2.0 step admits floor(8/2) + 1 = 5 ticks, exactly at the cap.
        (8.0, 5, 2.0),
        # 9 mm: floor(9/2) + 1 = 5 ticks, so 2.0 is still admissible.
        (9.0, 5, 2.0),
        # Narrow span: 0.05 gives floor(6) + 1 = 7 ticks, so 0.1 (4 ticks) is picked.
        (0.3, 5, 0.1),
        # Tighter cap forces a coarser step: 2.5 gives 4 ticks over 8 mm, 5.0 gives 2.
        (8.0, 3, 5.0),
        # Sub-decade span exercises the negative-exponent branch.
        (0.04, 5, 0.01),
    ],
)
def test_nice_tick_step_expected_values(span, max_ticks, expected):
    assert nice_tick_step(span, max_ticks=max_ticks) == pytest.approx(expected)


@pytest.mark.parametrize("span", [0.05, 0.4, 1.0, 3.7, 8.2, 12.5, 47.0])
@pytest.mark.parametrize("max_ticks", [3, 4, 5, 6])
def test_nice_tick_step_respects_max_ticks(span, max_ticks):
    """The step must keep the tick count within the cap for any axis offset."""
    step = nice_tick_step(span, max_ticks=max_ticks)
    # Worst case: the first tick sits on the lower limit, so ticks = floor(span/step) + 1.
    n_ticks = math.floor(span / step + 1e-9) + 1
    assert n_ticks <= max_ticks


@pytest.mark.parametrize("span", [0.05, 0.4, 1.0, 3.7, 8.2, 12.5, 47.0])
def test_nice_tick_step_uses_nice_mantissa(span):
    step = nice_tick_step(span, max_ticks=5)
    mantissa = step / 10.0 ** math.floor(math.log10(step))
    assert min(abs(mantissa - m) for m in (1.0, 2.0, 2.5, 5.0)) < 1e-9


@pytest.mark.parametrize("bad_span", [0.0, -1.0, float("nan"), float("inf")])
def test_nice_tick_step_rejects_invalid_span(bad_span):
    with pytest.raises(ValueError):
        nice_tick_step(bad_span)


def test_nice_tick_step_rejects_single_tick():
    with pytest.raises(ValueError):
        nice_tick_step(5.0, max_ticks=1)
