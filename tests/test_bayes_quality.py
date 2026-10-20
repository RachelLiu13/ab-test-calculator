import pytest

from abtest.bayesian import beta_binomial
from abtest.quality import srm_check


def test_identical_arms_give_coin_flip():
    r = beta_binomial(100, 1000, 100, 1000)
    assert r.prob_b_better == pytest.approx(0.5, abs=0.01)
    assert r.loss_choose_a == pytest.approx(r.loss_choose_b, rel=0.05)


def test_clear_winner():
    r = beta_binomial(100, 2000, 200, 2000)
    assert r.prob_b_better > 0.999
    assert r.lift_low > 0.5


def test_srm_flags_skewed_split_only():
    assert srm_check(50_000, 50_100)["mismatch"] is False
    assert srm_check(50_000, 51_500)["mismatch"] is True
