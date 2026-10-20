import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

from abtest.frequentist import sample_size_per_group, two_proportion_ztest


def test_sample_size_close_to_statsmodels():
    # statsmodels uses Cohen's h (arcsine transform); the two approximations agree within ~1%
    from statsmodels.stats.proportion import proportion_effectsize
    h = proportion_effectsize(0.11, 0.10)
    ref = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8, alternative="two-sided")
    assert sample_size_per_group(0.10, 0.10) == pytest.approx(ref, rel=0.01)


def test_smaller_effects_need_more_visitors():
    assert sample_size_per_group(0.05, 0.05) > sample_size_per_group(0.05, 0.20)


def test_ztest_matches_statsmodels():
    r = two_proportion_ztest(200, 2000, 250, 2000)
    z, p = proportions_ztest([250, 200], [2000, 2000])
    assert r.z == pytest.approx(z)
    assert r.p_value == pytest.approx(p)
    assert r.significant and r.ci_low > 0


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        two_proportion_ztest(10, 5, 1, 5)
    with pytest.raises(ValueError):
        sample_size_per_group(0.9, 0.2)
