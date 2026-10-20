"""Sample ratio mismatch (SRM): did traffic split the way the experiment was designed to?

A split that is off by more than chance (e.g. 50.8% / 49.2% on 100k users) usually means a bug in
assignment or logging, and the test result should not be trusted until it is explained.
"""
from scipy.stats import chisquare


def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5, threshold: float = 0.001) -> dict:
    total = n_a + n_b
    expected = [total * expected_share_a, total * (1 - expected_share_a)]
    stat, p = chisquare([n_a, n_b], f_exp=expected)
    return {"observed_share_a": n_a / total, "expected_share_a": expected_share_a, "chi2": float(stat),
            "p_value": float(p), "mismatch": bool(p < threshold)}
