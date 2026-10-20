"""Bayesian read-out with Beta-Binomial posteriors: 'how likely is B better, and what do we risk?'"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BayesResult:
    prob_b_better: float        # P(rate_B > rate_A | data)
    expected_lift: float        # posterior mean of (rate_B / rate_A - 1)
    lift_low: float             # 95% credible interval for the relative lift
    lift_high: float
    loss_choose_b: float        # expected conversion-rate loss if we ship B and A was better
    loss_choose_a: float


def beta_binomial(conv_a: int, n_a: int, conv_b: int, n_b: int, prior: tuple[float, float] = (1, 1),
                  draws: int = 200_000, seed: int = 0) -> BayesResult:
    """Monte Carlo over independent Beta(prior + successes, prior + failures) posteriors."""
    rng = np.random.default_rng(seed)
    a = rng.beta(prior[0] + conv_a, prior[1] + n_a - conv_a, draws)
    b = rng.beta(prior[0] + conv_b, prior[1] + n_b - conv_b, draws)
    lift = b / a - 1
    return BayesResult(
        prob_b_better=float((b > a).mean()),
        expected_lift=float(lift.mean()),
        lift_low=float(np.quantile(lift, 0.025)),
        lift_high=float(np.quantile(lift, 0.975)),
        loss_choose_b=float(np.maximum(a - b, 0).mean()),
        loss_choose_a=float(np.maximum(b - a, 0).mean()),
    )
