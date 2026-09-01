"""Setting-comparison figure stems must record the levels held fixed in the sweep.

``plot_settings_comparison`` legends name only the parameter that varies, so an agitation
sweep at 55 L min-1 and one at 90 L min-1 produce visually indistinguishable figures. The
stem is the only place the fixed levels are written down, and it is derived from the same
settings list that is plotted so the two cannot disagree.
"""

import pytest

from klarity.viz import setting_comparison_stem

RPM_SWEEP = [f"{rpm} rpm 55 lmin 000 xanthan" for rpm in (75, 100, 125, 150)]
LMIN_SWEEP = [f"100 rpm {lmin} lmin 025 xanthan" for lmin in (45, 55, 70, 90)]


def test_rpm_sweep_records_the_fixed_aeration_rate():
    assert setting_comparison_stem(RPM_SWEEP) == "rpm_at_55_lmin_000_xanthan"


def test_lmin_sweep_records_the_fixed_agitation_rate():
    assert setting_comparison_stem(LMIN_SWEEP) == "lmin_at_100_rpm_025_xanthan"


def test_prefix_and_suffix_wrap_the_stem():
    stem = setting_comparison_stem(LMIN_SWEEP, prefix="holdup_area_", suffix="_all_aeration")
    assert stem == "holdup_area_lmin_at_100_rpm_025_xanthan_all_aeration"


def test_same_sweep_at_different_fixed_levels_gets_different_stems():
    """Distinct parameter slices must produce distinct filenames."""
    at_55 = setting_comparison_stem([f"{rpm} rpm 55 lmin 000 xanthan" for rpm in (75, 150)])
    at_90 = setting_comparison_stem([f"{rpm} rpm 90 lmin 000 xanthan" for rpm in (75, 150)])
    assert at_55 != at_90


@pytest.mark.parametrize("xanthan", ["000", "0125", "025"])
def test_xanthan_level_is_kept_verbatim(xanthan):
    """'0125' and '025' are directory-level identifiers, not numbers to be reformatted."""
    stem = setting_comparison_stem([f"{rpm} rpm 55 lmin {xanthan} xanthan" for rpm in (75, 150)])
    assert stem == f"rpm_at_55_lmin_{xanthan}_xanthan"


def test_stem_carries_no_dot():
    """Callers append the extension with Path.with_suffix, which truncates at the last dot."""
    assert "." not in setting_comparison_stem(RPM_SWEEP)


def test_rejects_a_single_setting():
    with pytest.raises(ValueError, match="at least two settings"):
        setting_comparison_stem(["100 rpm 55 lmin 000 xanthan"])


def test_rejects_mixed_xanthan():
    with pytest.raises(ValueError, match="xanthan concentration must be held constant"):
        setting_comparison_stem(["100 rpm 55 lmin 000 xanthan", "100 rpm 55 lmin 025 xanthan"])


def test_rejects_two_varying_parameters():
    """A stem cannot say what is on the x axis if both rpm and aeration move."""
    with pytest.raises(ValueError, match="exactly one"):
        setting_comparison_stem(["75 rpm 45 lmin 000 xanthan", "150 rpm 90 lmin 000 xanthan"])


def test_rejects_settings_that_do_not_vary():
    with pytest.raises(ValueError, match="exactly one"):
        setting_comparison_stem(["100 rpm 55 lmin 000 xanthan"] * 2)
