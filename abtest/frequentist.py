"""Classical A/B test math for conversion rates (two independent proportions)."""
from dataclasses import dataclass
from math import ceil, sqrt

from scipy.stats import norm


def sample_size_per_group(baseline: float, mde_relative: float, alpha: float = 0.05, power: float = 0.80,
                          two_sided: bool = True) -> int:
    """Visitors needed in EACH group to detect a relative lift of `mde_relative` over `baseline`.

    Uses the standard normal approximation with unpooled variances:
        n = (z_alpha * sqrt(2 p_bar q_bar) + z_beta * sqrt(p1 q1 + p2 q2))^2 / (p2 - p1)^2
    """
    if not 0 < baseline < 1:
        raise ValueError("baseline must be between 0 and 1")
    p1, p2 = baseline, baseline * (1 + mde_relative)
    if not 0 < p2 < 1 or p1 == p2:
        raise ValueError("baseline x (1 + lift) must be between 0 and 1 and differ from baseline")
    z_a = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
    z_b = norm.ppf(power)
    p_bar = (p1 + p2) / 2
    n = (z_a * sqrt(2 * p_bar * (1 - p_bar)) + z_b * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2
    return ceil(n)


@dataclass(frozen=True)
class ZTestResult:
    rate_a: float
    rate_b: float
    lift: float                 # relative lift of B over A
    diff: float                 # absolute difference B - A
    ci_low: float               # confidence interval for the absolute difference
    ci_high: float
    z: float
    p_value: float
    significant: bool


def two_proportion_ztest(conv_a: int, n_a: int, conv_b: int, n_b: int, alpha: float = 0.05) -> ZTestResult:
    """Two-sided z-test for a difference in conversion rates, with a Wald CI for the difference."""
    if min(n_a, n_b) <= 0 or not (0 <= conv_a <= n_a and 0 <= conv_b <= n_b):
        raise ValueError("conversions must be between 0 and the number of visitors")
    pa, pb = conv_a / n_a, conv_b / n_b
    pooled = (conv_a + conv_b) / (n_a + n_b)
    se_pooled = sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se_pooled if se_pooled > 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    se = sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    half = norm.ppf(1 - alpha / 2) * se
    return ZTestResult(pa, pb, (pb - pa) / pa if pa > 0 else float("nan"), pb - pa, pb - pa - half, pb - pa + half,
                       z, p, p < alpha)
