"""A/B Test Calculator - plan, analyze and sanity-check conversion experiments.

Run:  streamlit run app.py
"""
from math import ceil

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from abtest.bayesian import beta_binomial
from abtest.frequentist import sample_size_per_group, two_proportion_ztest
from abtest.quality import srm_check

st.set_page_config(page_title="A/B Test Calculator", page_icon="🧪", layout="centered")
st.title("🧪 A/B Test Calculator")
st.caption("Plan how long to run a conversion test, read out the result two ways, and check the traffic split.")

plan, analyze, srm = st.tabs(["1 · Plan", "2 · Analyze", "3 · Sanity check"])

with plan:
    st.subheader("How many visitors do I need?")
    c1, c2 = st.columns(2)
    baseline = c1.number_input("Current conversion rate (%)", 0.1, 99.0, 5.0, 0.1) / 100
    mde = c2.number_input("Smallest relative lift worth detecting (%)", 1.0, 200.0, 10.0, 1.0) / 100
    c3, c4, c5 = st.columns(3)
    alpha = c3.selectbox("Significance level", [0.01, 0.05, 0.10], index=1)
    power = c4.selectbox("Power", [0.80, 0.90], index=0, format_func=lambda v: f"{v:.0%}")
    daily = c5.number_input("Visitors per day (both groups)", 10, 10_000_000, 2000, 100)
    try:
        n = sample_size_per_group(baseline, mde, alpha, power)
        days = ceil(2 * n / daily)
        m1, m2, m3 = st.columns(3)
        m1.metric("Visitors per group", f"{n:,}")
        m2.metric("Total visitors", f"{2 * n:,}")
        m3.metric("Days to run", f"{days:,}", help="Round up to whole weeks so every weekday is covered equally.")
        st.info(f"To detect a change from **{baseline:.2%}** to **{baseline * (1 + mde):.2%}** "
                f"with {power:.0%} power at α = {alpha}, run for about **{max(days, 7)} days** "
                f"({ceil(max(days, 7) / 7)} full weeks). Decide this in advance and don't stop early when the result looks good.")
        lifts = np.linspace(max(0.02, mde / 3), mde * 3, 40)
        ns = [sample_size_per_group(baseline, l, alpha, power) for l in lifts]
        fig = go.Figure(go.Scatter(x=lifts * 100, y=ns, mode="lines", line=dict(color="#2a78d6", width=2),
                                   hovertemplate="lift %{x:.1f}%: %{y:,} per group<extra></extra>"))
        fig.update_layout(height=280, margin=dict(l=8, r=8, t=30, b=8), title="Smaller lifts need far more traffic",
                          xaxis_title="Relative lift to detect (%)", yaxis_title="Visitors per group",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    except ValueError as e:
        st.error(str(e))

with analyze:
    st.subheader("Did B beat A?")
    c1, c2 = st.columns(2)
    n_a = c1.number_input("Visitors A", 1, 100_000_000, 10_000)
    conv_a = c1.number_input("Conversions A", 0, 100_000_000, 500)
    n_b = c2.number_input("Visitors B", 1, 100_000_000, 10_000)
    conv_b = c2.number_input("Conversions B", 0, 100_000_000, 570)
    try:
        z = two_proportion_ztest(conv_a, n_a, conv_b, n_b)
        b = beta_binomial(conv_a, n_a, conv_b, n_b)
        m1, m2, m3 = st.columns(3)
        m1.metric("Conversion A", f"{z.rate_a:.2%}")
        m2.metric("Conversion B", f"{z.rate_b:.2%}", delta=f"{z.lift:+.1%} relative")
        m3.metric("P(B better than A)", f"{b.prob_b_better:.1%}")
        st.markdown("**Frequentist (two-proportion z-test)**")
        st.markdown(f"- p-value **{z.p_value:.4f}** → {'statistically significant' if z.significant else 'not significant'} at α = 0.05\n"
                    f"- 95% CI for the difference: **{z.ci_low:+.2%} to {z.ci_high:+.2%}** (percentage points)")
        st.markdown("**Bayesian (Beta-Binomial, flat prior)**")
        st.markdown(f"- Probability B is better: **{b.prob_b_better:.1%}**\n"
                    f"- Expected relative lift **{b.expected_lift:+.1%}**, 95% credible interval {b.lift_low:+.1%} to {b.lift_high:+.1%}\n"
                    f"- Expected loss if you ship B: **{b.loss_choose_b:.3%}** of conversion rate "
                    f"(vs {b.loss_choose_a:.3%} if you keep A)")
        verdict = ("Ship B." if z.significant and z.diff > 0 else
                   "Keep A." if z.significant and z.diff < 0 else
                   "No clear winner yet - keep running to the planned sample size, or accept that the effect is small.")
        st.success(f"**Read-out:** {verdict}")
    except ValueError as e:
        st.error(str(e))

with srm:
    st.subheader("Did traffic split as designed?")
    c1, c2, c3 = st.columns(3)
    sa = c1.number_input("Visitors in A", 1, 100_000_000, 50_000, key="srm_a")
    sb = c2.number_input("Visitors in B", 1, 100_000_000, 51_500, key="srm_b")
    share = c3.number_input("Designed share for A (%)", 1.0, 99.0, 50.0) / 100
    r = srm_check(sa, sb, share)
    if r["mismatch"]:
        st.error(f"⚠️ Sample ratio mismatch: A got {r['observed_share_a']:.2%} of traffic vs {share:.0%} designed "
                 f"(χ² p = {r['p_value']:.2e}). Find the assignment or logging bug before trusting the result.")
    else:
        st.success(f"✅ Split looks fine: A got {r['observed_share_a']:.2%} vs {share:.0%} designed (p = {r['p_value']:.3f}).")

st.caption("Formulas: normal-approximation sample size and z-test; Beta(1,1) prior; SRM via χ² with p < 0.001. "
           "Source: github.com/RachelLiu13/ab-test-calculator")
