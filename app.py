"""
Streamlit app for the CDNOW customer value project.

Reads the scored customer file produced by clv_pipeline.py (outputs/customer_scores.csv),
so it starts fast and needs no model fitting. Run locally with:  streamlit run app.py
"""
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

NAVY, TEAL, ORANGE, GREY = "#1f3a5f", "#2a9d8f", "#e9843b", "#8d99ae"

st.set_page_config(page_title="CDNOW customer value", page_icon=None, layout="wide")


@st.cache_data
def load():
    c = pd.read_csv(os.path.join(OUT, "customer_scores.csv"), index_col=0)
    comp = pd.read_csv(os.path.join(OUT, "model_comparison.csv"))
    with open(os.path.join(OUT, "metrics.json")) as f:
        m = json.load(f)
    return c, comp, m


cust, comp, metrics = load()
seg_names = cust.groupby("segment").segment_name.first().to_dict()
seg_label = {k: f"{k}  {v}" for k, v in seg_names.items()}

st.title("Customer segmentation, CLV and targeting economics")
st.write(
    "Real purchase history from CDNOW, an online music retailer: "
    f"{metrics['modelled_customers']:,} customers and {metrics['orders']:,} orders from 1997 to mid-1998. "
    "Models are fit on 1997 and checked against the next six months, which the models never saw."
)

tab_over, tab_seg, tab_target, tab_test, tab_limits = st.tabs(
    ["Forecast accuracy", "Segments", "Targeting calculator", "Test sizing", "Where it falls short"]
)

# ------------------------------------------------------------------ forecast accuracy
with tab_over:
    a = metrics["aggregate_holdout"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders, predicted", f"{int(a['pred_orders']):,}", f"{a['orders_error_pct']:+.1f}% vs actual")
    c2.metric("Orders, actual", f"{a['actual_orders']:,}")
    c3.metric("Spend, predicted", f"${a['pred_spend']:,.0f}", f"{a['spend_error_pct']:+.1f}% vs actual")
    c4.metric("Spend, actual", f"${a['actual_spend']:,.0f}")

    st.subheader("Ranking customers")
    st.write(
        "Contacting customers in order of predicted spend, how much of the actual holdout spend do you reach? "
        "The three approaches land close together, so the choice between them rests on explainability, not accuracy."
    )
    models = {
        "BG/NBD + Gamma-Gamma": "pred_spend",
        "Gradient boosting": "gb_spend",
        "Past rate carried forward": "runrate_orders",
    }
    rows = []
    n = len(cust)
    for name, col in models.items():
        o = cust.sort_values(col, ascending=False)
        cum = o.spend_hold.cumsum().values / o.spend_hold.sum()
        pts = np.linspace(0, n - 1, 101).astype(int)
        rows.append(pd.DataFrame({"pct_contacted": np.linspace(0, 100, 101),
                                  "pct_spend": 100 * cum[pts], "model": name}))
    gains = pd.concat(rows)
    chart = (
        alt.Chart(gains)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("pct_contacted:Q", title="% of customers contacted (ranked by predicted spend)"),
            y=alt.Y("pct_spend:Q", title="% of actual holdout spend reached"),
            color=alt.Color("model:N", scale=alt.Scale(range=[NAVY, TEAL, GREY]), legend=alt.Legend(title=None)),
        )
        .properties(height=340)
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(
        comp.rename(columns={
            "model": "Model", "AUC_any_purchase": "AUC (bought at all)",
            "spearman_vs_actual_spend": "Spearman vs actual spend",
            "top10pct_spend_capture": "Spend share in top 10%"}),
        hide_index=True, width="stretch",
    )
    st.caption(
        "The Spearman column favours the naive rule, but 77% of customers spent nothing in the holdout, "
        "so that metric mostly measures ties."
    )

# ------------------------------------------------------------------ segments
with tab_seg:
    g = cust.groupby("segment")
    prof = pd.DataFrame({
        "Segment": [seg_label[s] for s in g.size().index],
        "Customers": g.size().values,
        "% of customers": 100 * g.size().values / len(cust),
        "% of holdout spend": 100 * g.spend_hold.sum().values / cust.spend_hold.sum(),
        "Avg 1997 spend ($)": g.spend_cal.mean().values,
        "Avg predicted 6-mo spend ($)": g.pred_spend.mean().values,
        "Avg actual 6-mo spend ($)": g.spend_hold.mean().values,
        "% who bought Jan-Jun 1998": 100 * g.bought_hold.mean().values,
    })
    st.write(
        "Five groups from recency, frequency and spend (log-scaled, k-means). Silhouette scores for four to seven "
        "groups are within 0.04 of each other, so five is a usability choice."
    )
    long = prof.melt(id_vars="Segment", value_vars=["% of customers", "% of holdout spend"],
                     var_name="measure", value_name="pct")
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("Segment:N", sort=None, axis=alt.Axis(labelAngle=0, labelLimit=140), title=None),
            xOffset="measure:N",
            y=alt.Y("pct:Q", title="%"),
            color=alt.Color("measure:N", scale=alt.Scale(range=[GREY, NAVY]), legend=alt.Legend(title=None)),
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(
        prof.style.format({
            "Customers": "{:,.0f}", "% of customers": "{:.1f}", "% of holdout spend": "{:.1f}",
            "Avg 1997 spend ($)": "{:.0f}", "Avg predicted 6-mo spend ($)": "{:.1f}",
            "Avg actual 6-mo spend ($)": "{:.1f}", "% who bought Jan-Jun 1998": "{:.1f}"}),
        hide_index=True, width="stretch",
    )

# ------------------------------------------------------------------ targeting calculator
with tab_target:
    st.write(
        "A campaign pays back only if it lifts spend by more than the contact costs. "
        "The break-even lift for each decile of predicted spend comes from the model. "
        "The cost, margin and lift below are your assumptions to change; the data has no campaign results."
    )
    s1, s2, s3 = st.columns(3)
    cost = s1.slider("Cost per contact ($)", 0.10, 3.00, 0.60, 0.05)
    margin = s2.slider("Contribution margin (%)", 10, 60, 35, 1) / 100
    lift = s3.slider("Assumed relative lift in 6-month spend (%)", 1, 40, 10, 1) / 100

    dec = cust.groupby("decile").agg(
        customers=("pred_spend", "size"),
        avg_pred=("pred_spend", "mean"),
        avg_actual=("spend_hold", "mean"),
        purchase_rate=("bought_hold", "mean"),
    )
    dec["breakeven"] = cost / (dec.avg_pred * margin)
    dec["clears"] = dec.breakeven <= lift

    gain = lift * cust.spend_hold * margin - cost
    contact_all = float(gain.sum())
    rule_net = float(gain[cust.decile.isin(dec.index[dec.clears])].sum())
    order = cust.sort_values("pred_spend", ascending=False).index
    cum = gain.loc[order].cumsum().values
    best_i = int(np.argmax(cum))

    def usd(v):
        return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"

    n_clear = int(dec.clears.sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Contact everyone", usd(contact_all))
    m2.metric(f"Contact the {n_clear} {'decile that clears' if n_clear == 1 else 'deciles that clear'} break-even",
              usd(rule_net))
    m3.metric("Best depth, found with hindsight", usd(float(cum[best_i])),
              f"top {100 * (best_i + 1) / len(cust):.0f}% of customers", delta_color="off")
    st.caption(
        "Net value over the six-month window, applying the assumed lift to each customer's actual holdout spend. "
        "The middle figure uses a rule you could set before the campaign; the right-hand figure picks the best "
        "depth after seeing the outcome, so treat it as a ceiling."
    )

    cdf = pd.DataFrame({"decile": dec.index.astype(str), "breakeven": 100 * dec.breakeven.values,
                        "clears": np.where(dec.clears.values, "Clears assumed lift", "Does not clear")})
    bars = (
        alt.Chart(cdf)
        .mark_bar()
        .encode(
            x=alt.X("decile:N", sort=[str(i) for i in range(1, 11)], title="Predicted-spend decile (1 = highest)",
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("breakeven:Q", scale=alt.Scale(type="log"), title="Break-even lift in spend (%, log scale)"),
            color=alt.Color("clears:N", scale=alt.Scale(domain=["Clears assumed lift", "Does not clear"],
                                                        range=[TEAL, GREY]), legend=alt.Legend(title=None)),
        )
    )
    rule = alt.Chart(pd.DataFrame({"y": [100 * lift]})).mark_rule(color=ORANGE, strokeDash=[5, 4], size=2).encode(y="y:Q")
    st.altair_chart((bars + rule).properties(height=320), width="stretch")

    curve = pd.DataFrame({
        "pct_contacted": 100 * (np.arange(1, len(cum) + 1)) / len(cum),
        "net_value": cum,
    }).iloc[::50]
    line = (
        alt.Chart(curve)
        .mark_line(color=NAVY, strokeWidth=2.5)
        .encode(x=alt.X("pct_contacted:Q", title="% of customers contacted (ranked by predicted spend)"),
                y=alt.Y("net_value:Q", title="Cumulative net value ($)"))
        .properties(height=300)
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#cccccc").encode(y="y:Q")
    st.altair_chart(zero + line, width="stretch")

    st.dataframe(
        pd.DataFrame({
            "Decile": dec.index,
            "Customers": dec.customers.values,
            "Avg predicted spend ($)": dec.avg_pred.values,
            "Avg actual spend ($)": dec.avg_actual.values,
            "Bought in holdout (%)": 100 * dec.purchase_rate.values,
            "Break-even lift (%)": 100 * dec.breakeven.values,
        }).style.format({"Customers": "{:,.0f}", "Avg predicted spend ($)": "{:.2f}",
                         "Avg actual spend ($)": "{:.2f}", "Bought in holdout (%)": "{:.1f}",
                         "Break-even lift (%)": "{:.1f}"}),
        hide_index=True, width="stretch",
    )

# ------------------------------------------------------------------ test sizing
with tab_test:
    st.write(
        "Before spending on a campaign, size the holdout test that would confirm the lift is real. "
        "Baselines come from the real six-month holdout for the chosen segment."
    )
    t1, t2 = st.columns(2)
    choice = t1.selectbox("Segment", ["All customers"] + [seg_label[s] for s in sorted(seg_label)])
    metric = t2.radio("Measure the test on", ["Purchase rate", "Spend per customer"], horizontal=True)
    t3, t4, t5 = st.columns(3)
    rel = t3.slider("Relative lift to detect (%)", 5, 50, 20, 5) / 100
    power = t4.select_slider("Power", options=[0.7, 0.8, 0.9], value=0.8)
    alpha = t5.select_slider("Significance level", options=[0.01, 0.05, 0.10], value=0.05)

    if choice == "All customers":
        sub = cust
    else:
        sub = cust[cust.segment == choice.split()[0]]

    if metric == "Purchase rate":
        p0 = float(sub.bought_hold.mean())
        es = proportion_effectsize(min(p0 * (1 + rel), 0.999), p0)
        n_arm = int(np.ceil(NormalIndPower().solve_power(effect_size=es, alpha=alpha, power=power,
                                                         ratio=1.0, alternative="two-sided")))
        base = f"Baseline purchase rate in the holdout: {100 * p0:.1f}%"
    else:
        mu, sd = float(sub.spend_hold.mean()), float(sub.spend_hold.std())
        z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
        n_arm = int(np.ceil(2 * z ** 2 * sd ** 2 / ((rel * mu) ** 2)))
        base = f"Baseline spend per customer: ${mu:,.2f} (standard deviation ${sd:,.2f})"

    r1, r2, r3 = st.columns(3)
    r1.metric("Customers per arm", f"{n_arm:,}")
    r2.metric("Total for two arms", f"{2 * n_arm:,}")
    r3.metric("Customers in this group", f"{len(sub):,}")
    st.caption(base)
    if 2 * n_arm <= len(sub):
        st.success("This group is large enough to run the test on its own.")
    else:
        st.warning(f"This group has {len(sub):,} customers and the test needs {2 * n_arm:,}. "
                   "Pool it with another segment, accept a larger detectable lift, or test on purchase rate.")

# ------------------------------------------------------------------ limits
with tab_limits:
    st.write("Three places the model is weaker than the headline numbers suggest.")

    st.subheader("It writes off quiet repeat buyers too early")
    rb = cust[cust.frequency > 0].copy()
    rb["Probability still active (model)"] = pd.cut(
        rb.p_alive, [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["under 0.2", "0.2 to 0.4", "0.4 to 0.6", "0.6 to 0.8", "over 0.8"])
    ab = rb.groupby("Probability still active (model)", observed=True).agg(
        Customers=("pred_orders", "size"), Predicted=("pred_orders", "mean"),
        Actual=("orders_hold", "mean"), Bought=("bought_hold", "mean")).reset_index()
    ab["Bought"] = 100 * ab["Bought"]
    ab = ab.rename(columns={"Predicted": "Predicted orders", "Actual": "Actual orders", "Bought": "Bought in holdout (%)"})
    st.dataframe(ab.style.format({"Customers": "{:,.0f}", "Predicted orders": "{:.3f}",
                                  "Actual orders": "{:.3f}", "Bought in holdout (%)": "{:.1f}"}),
                 hide_index=True, width="stretch")
    st.caption(
        "Repeat buyers rated under 0.2 placed about three times the predicted orders, and 19% bought at all, "
        "against 8% for lapsed one-time buyers. A low score should not be read as churned."
    )

    st.subheader("It misses at both ends of purchase frequency")
    fb = cust.groupby("freq_bin", observed=True).agg(
        Customers=("pred_orders", "size"), Predicted=("pred_orders", "mean"), Actual=("orders_hold", "mean"))
    order_bins = ["0", "1", "2", "3", "4", "5-6", "7+"]
    fb = fb.reindex([b for b in order_bins if b in fb.index]).reset_index().rename(
        columns={"freq_bin": "Repeat orders in 1997", "Predicted": "Predicted orders", "Actual": "Actual orders"})
    st.dataframe(fb.style.format({"Customers": "{:,.0f}", "Predicted orders": "{:.3f}", "Actual orders": "{:.3f}"}),
                 hide_index=True, width="stretch")

    st.subheader("Probability still active is always 1.0 for one-time buyers")
    st.write(
        "Under BG/NBD a customer cannot drop out before a second purchase, so the metric says nothing about "
        "whether a one-time buyer is gone. The app and the write-up do not use it for them."
    )
    st.caption(
        "Scope: one retailer, a 1997 to 1998 cohort, no channel or campaign fields. "
        "This is not attribution or a marketing mix model, and nothing here measures incrementality."
    )
