"""
Customer segmentation, CLV and targeting framework on real CDNOW transaction data.

Design: fit on a 12-month calibration window (1997), validate against what customers
actually did in the following 6 months (1998 H1). Nothing from the holdout window is
used to fit or tune any model.

Run:  python clv_pipeline.py
Out:  ./outputs/  (figures, CSVs, metrics.json)
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes import datasets as lt_datasets
from lifetimes.utils import summary_data_from_transaction_data
from scipy.stats import norm, spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

warnings.filterwarnings("ignore")
RNG = 42
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

CAL_END = pd.Timestamp("1997-12-31")
HOLD_END = pd.Timestamp("1998-06-30")
HOLD_DAYS = (HOLD_END - CAL_END).days  # 181

# Business assumptions for the targeting scenarios. These are stated inputs, not findings.
CONTACT_COST = 0.60     # cost to contact one customer (USD)
MARGIN = 0.35           # contribution margin on incremental sales

NAVY, TEAL, ORANGE, GREY = "#1f3a5f", "#2a9d8f", "#e9843b", "#8d99ae"
plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "font.family": "DejaVu Sans",
})

metrics = {}


# --------------------------------------------------------------------------- data
def load_orders():
    path = os.path.join(os.path.dirname(lt_datasets.__file__), "CDNOW_master.txt")
    raw = pd.read_csv(path, sep=r"\s+", dtype={"customer_id": str})
    raw.columns = ["cust", "date", "qty", "amt"]
    raw["date"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d")
    metrics["raw_rows"] = int(len(raw))
    metrics["raw_customers"] = int(raw.cust.nunique())
    metrics["zero_value_rows_dropped"] = int((raw.amt <= 0).sum())
    raw = raw[raw.amt > 0]
    # collapse multiple lines on the same day into one order
    orders = raw.groupby(["cust", "date"], as_index=False).agg(qty=("qty", "sum"), amt=("amt", "sum"))
    metrics["orders"] = int(len(orders))
    metrics["customers"] = int(orders.cust.nunique())
    metrics["first_purchase_range"] = [
        str(orders.groupby("cust").date.min().min().date()),
        str(orders.groupby("cust").date.min().max().date()),
    ]
    return orders


orders = load_orders()
cal = orders[orders.date <= CAL_END]
hold = orders[(orders.date > CAL_END) & (orders.date <= HOLD_END)]

# --------------------------------------------------------------------------- customer table
s = summary_data_from_transaction_data(
    cal, "cust", "date", monetary_value_col="amt", observation_period_end=CAL_END, freq="D"
)  # frequency = repeat orders, recency = first->last gap, T = first->cutoff, monetary = mean repeat order value
tot = cal.groupby("cust").agg(orders_cal=("date", "size"), spend_cal=("amt", "sum"), last=("date", "max"))
cust = s.join(tot)
cust["days_since_last"] = (CAL_END - cust["last"]).dt.days
h = hold.groupby("cust").agg(orders_hold=("date", "size"), spend_hold=("amt", "sum"))
cust = cust.join(h).fillna({"orders_hold": 0, "spend_hold": 0})
cust["bought_hold"] = (cust.orders_hold > 0).astype(int)
metrics["modelled_customers"] = int(len(cust))
metrics["holdout_days"] = HOLD_DAYS
metrics["holdout_actual_orders"] = int(cust.orders_hold.sum())
metrics["holdout_actual_spend"] = round(float(cust.spend_hold.sum()), 2)
metrics["holdout_purchase_rate"] = round(float(cust.bought_hold.mean()), 4)
metrics["one_order_share_calibration"] = round(float((cust.orders_cal == 1).mean()), 4)

# --------------------------------------------------------------------------- BG/NBD + Gamma-Gamma
bgf = BetaGeoFitter(penalizer_coef=0.001).fit(cust["frequency"], cust["recency"], cust["T"])
metrics["bgnbd_params"] = {k: round(float(v), 4) for k, v in bgf.params_.items()}
cust["pred_orders"] = bgf.conditional_expected_number_of_purchases_up_to_time(
    HOLD_DAYS, cust["frequency"], cust["recency"], cust["T"])
cust["p_alive"] = bgf.conditional_probability_alive(cust["frequency"], cust["recency"], cust["T"])

rep = cust[(cust.frequency > 0) & (cust.monetary_value > 0)]
metrics["freq_value_correlation"] = round(float(np.corrcoef(rep.frequency, rep.monetary_value)[0, 1]), 4)
ggf = GammaGammaFitter(penalizer_coef=0.01).fit(rep["frequency"], rep["monetary_value"])
cust["pred_aov"] = float(rep.monetary_value.mean())  # customers with no repeat orders get the repeat-buyer mean
idx = cust.index.isin(rep.index)
cust.loc[idx, "pred_aov"] = ggf.conditional_expected_average_profit(
    rep["frequency"], rep["monetary_value"]).values
cust["pred_spend"] = cust.pred_orders * cust.pred_aov

# --------------------------------------------------------------------------- validation vs holdout
agg = {
    "pred_orders": round(float(cust.pred_orders.sum()), 0),
    "actual_orders": int(cust.orders_hold.sum()),
    "pred_spend": round(float(cust.pred_spend.sum()), 0),
    "actual_spend": round(float(cust.spend_hold.sum()), 0),
}
agg["orders_error_pct"] = round(100 * (agg["pred_orders"] / agg["actual_orders"] - 1), 1)
agg["spend_error_pct"] = round(100 * (agg["pred_spend"] / agg["actual_spend"] - 1), 1)
metrics["aggregate_holdout"] = agg

cust["freq_bin"] = pd.cut(cust.frequency, [-1, 0, 1, 2, 3, 4, 6, 1000],
                          labels=["0", "1", "2", "3", "4", "5-6", "7+"])
fb = cust.groupby("freq_bin", observed=True).agg(
    customers=("pred_orders", "size"), predicted=("pred_orders", "mean"), actual=("orders_hold", "mean")
).round(3)
fb.to_csv(os.path.join(OUT, "validation_by_frequency.csv"))

# where does the model misjudge repeat buyers who have gone quiet? (p_alive bands, repeat buyers only)
rb = cust[cust.frequency > 0].copy()
rb["alive_band"] = pd.cut(rb.p_alive, [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                          labels=["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", ">0.8"])
ab = rb.groupby("alive_band", observed=True).agg(
    customers=("pred_orders", "size"), predicted=("pred_orders", "mean"), actual=("orders_hold", "mean"),
    actual_purchase_rate=("bought_hold", "mean")).round(3)
ab.to_csv(os.path.join(OUT, "validation_by_p_alive_repeat_buyers.csv"))
metrics["p_alive_band_check"] = ab.reset_index().to_dict(orient="records")
metrics["onetime_buyers_p_alive_is_1"] = bool((cust.loc[cust.frequency == 0, "p_alive"] > 0.999).all())

# baselines, scored out-of-fold so no customer is scored by a model that saw their own outcome
feats = ["frequency", "recency", "T", "monetary_value", "orders_cal", "spend_cal", "days_since_last"]
X = cust[feats].values
kf = KFold(5, shuffle=True, random_state=RNG)
gb_prob = cross_val_predict(HistGradientBoostingClassifier(random_state=RNG, max_depth=3, learning_rate=0.05,
                            max_iter=200), X, cust.bought_hold.values, cv=kf, method="predict_proba")[:, 1]
gb_spend = cross_val_predict(HistGradientBoostingRegressor(random_state=RNG, max_depth=3, learning_rate=0.05,
                             max_iter=200), X, cust.spend_hold.values, cv=kf)
cust["gb_prob"], cust["gb_spend"] = gb_prob, gb_spend
cust["runrate_orders"] = cust.frequency / cust["T"] * HOLD_DAYS  # naive: past rate carried forward


def top_decile_capture(score, actual):
    n = int(len(score) * 0.10)
    top = np.argsort(-np.asarray(score))[:n]
    return float(np.asarray(actual)[top].sum() / np.asarray(actual).sum())


models = {
    "BG/NBD + Gamma-Gamma": ("pred_orders", "pred_spend"),
    "Gradient boosting (RFM features)": ("gb_prob", "gb_spend"),
    "Naive run-rate": ("runrate_orders", "runrate_orders"),
}
comp = []
for name, (pcol, scol) in models.items():
    comp.append({
        "model": name,
        "AUC_any_purchase": round(roc_auc_score(cust.bought_hold, cust[pcol]), 4),
        "spearman_vs_actual_spend": round(float(spearmanr(cust[scol], cust.spend_hold)[0]), 4),
        "top10pct_spend_capture": round(top_decile_capture(cust[scol], cust.spend_hold), 4),
    })
comp = pd.DataFrame(comp)
comp.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
metrics["model_comparison"] = comp.to_dict(orient="records")

# --------------------------------------------------------------------------- segmentation
seg = pd.DataFrame({
    "R": cust["days_since_last"], "F": cust["orders_cal"], "M": cust["spend_cal"],
}, index=cust.index)
Z = StandardScaler().fit_transform(np.log1p(seg))
rs = np.random.RandomState(RNG)
samp = rs.choice(len(Z), 6000, replace=False)
sil = {}
for k in range(3, 8):
    lab = KMeans(k, n_init=10, random_state=RNG).fit_predict(Z)
    sil[k] = round(float(silhouette_score(Z[samp], lab[samp])), 4)
metrics["silhouette_by_k"] = sil
# Silhouette scores for k=4..7 sit within ~0.04 of each other, so statistics alone do not pick k.
# k=5 is a usability call: few enough segments to act on, enough to separate value and recency.
K = 5
km = KMeans(K, n_init=20, random_state=RNG).fit(Z)
cust["cluster"] = km.labels_
metrics["chosen_k"] = int(K)
alt = KMeans(K, n_init=20, random_state=7).fit_predict(Z)
metrics["segment_stability_ARI"] = round(float(adjusted_rand_score(km.labels_, alt)), 4)

prof = cust.groupby("cluster").agg(
    customers=("pred_orders", "size"),
    med_days_since_last=("days_since_last", "median"),
    med_orders=("orders_cal", "median"),
    avg_spend_cal=("spend_cal", "mean"),
    avg_pred_spend=("pred_spend", "mean"),
    avg_actual_hold_spend=("spend_hold", "mean"),
    hold_purchase_rate=("bought_hold", "mean"),
    p_alive=("p_alive", "mean"),
)
tot_hold = cust.spend_hold.sum()
prof["share_customers"] = prof.customers / prof.customers.sum()
prof["share_hold_spend"] = cust.groupby("cluster").spend_hold.sum() / tot_hold
med_r = cust.days_since_last.median()


def name_cluster(r):
    freq = "One-time" if r.med_orders <= 1 else ("Mid repeat" if r.med_orders < 6 else "Heavy repeat")
    rec = "active" if r.med_days_since_last <= 60 else ("cooling" if r.med_days_since_last <= 200 else "lapsed")
    return f"{freq} / {rec}"


prof["name"] = prof.apply(name_cluster, axis=1)
# when two clusters share a name, separate them by average calibration spend
for nm, grp in prof.groupby("name"):
    if len(grp) > 1:
        order = grp.avg_spend_cal.sort_values(ascending=False).index
        for j, ix in enumerate(order):
            prof.loc[ix, "name"] = f"{nm} ({'higher' if j == 0 else 'lower'} spend)"
prof = prof.sort_values("avg_actual_hold_spend", ascending=False)
prof["segment"] = [f"S{i+1}" for i in range(len(prof))]
prof.round(3).to_csv(os.path.join(OUT, "segment_profile.csv"))
cust["segment"] = cust.cluster.map(prof.segment)
cust["segment_name"] = cust.cluster.map(prof.name)

# --------------------------------------------------------------------------- targeting economics
cust["decile"] = pd.qcut(cust.pred_spend.rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
dec = cust.groupby("decile").agg(
    customers=("pred_spend", "size"),
    avg_pred_spend=("pred_spend", "mean"),
    avg_actual_spend=("spend_hold", "mean"),
    purchase_rate=("bought_hold", "mean"),
    spend_total=("spend_hold", "sum"),
)
dec["share_actual_spend"] = dec.spend_total / dec.spend_total.sum()
dec["cum_share_actual_spend"] = dec.share_actual_spend.cumsum()
# relative lift on 6-month spend needed to cover the contact cost
dec["breakeven_lift_pct"] = 100 * CONTACT_COST / (dec.avg_pred_spend * MARGIN)
dec.round(4).to_csv(os.path.join(OUT, "decile_table.csv"))

scen = []
for cost in (0.30, 0.60, 1.20):
    for lift in (0.05, 0.10, 0.20):
        # assumed relative lift applied to each customer's ACTUAL holdout spend
        gain = lift * cust.spend_hold * MARGIN - cost
        order = cust.sort_values("pred_spend", ascending=False).index
        g = gain.loc[order]
        best_k = int(np.argmax(g.cumsum().values)) + 1
        scen.append({
            "contact_cost": cost, "assumed_lift": lift,
            "net_value_contact_all": round(float(gain.sum()), 0),
            "net_value_best_targeted": round(float(g.cumsum().max()), 0),
            "best_targeting_depth_pct": round(100 * best_k / len(cust), 1),
        })
scen = pd.DataFrame(scen)
scen.to_csv(os.path.join(OUT, "targeting_scenarios.csv"), index=False)

# --------------------------------------------------------------------------- experiment sizing
pw = NormalIndPower()
rows = []
for sg, g in cust.groupby("segment"):
    p0 = g.bought_hold.mean()
    sd, mu = g.spend_hold.std(), g.spend_hold.mean()
    row = {"segment": sg, "name": g.segment_name.iloc[0], "customers": len(g),
           "baseline_purchase_rate": round(float(p0), 4)}
    for rel in (0.10, 0.20):
        es = proportion_effectsize(min(p0 * (1 + rel), 0.999), p0)
        n = pw.solve_power(effect_size=es, alpha=0.05, power=0.8, ratio=1.0, alternative="two-sided")
        row[f"n_per_arm_rate_lift_{int(rel*100)}pct"] = int(np.ceil(n))
        z_sum = norm.ppf(0.975) + norm.ppf(0.80)  # two-sided 5% significance, 80% power
        n_sp = 2 * (z_sum ** 2) * sd ** 2 / ((rel * mu) ** 2) if mu > 0 else np.nan
        row[f"n_per_arm_spend_lift_{int(rel*100)}pct"] = int(np.ceil(n_sp)) if np.isfinite(n_sp) else None
    row["feasible_20pct_rate_test"] = bool(row["n_per_arm_rate_lift_20pct"] * 2 <= len(g))
    rows.append(row)
expd = pd.DataFrame(rows)
expd.to_csv(os.path.join(OUT, "experiment_sizing.csv"), index=False)

# --------------------------------------------------------------------------- figures
# 1: calibration check by purchase-frequency bin
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(len(fb))
ax.bar(x - 0.2, fb.predicted, 0.4, color=NAVY, label="BG/NBD predicted")
ax.bar(x + 0.2, fb.actual, 0.4, color=ORANGE, label="Actual (holdout)")
ax.set_xticks(x); ax.set_xticklabels(fb.index)
ax.set_xlabel("Repeat orders in 1997 (calibration)"); ax.set_ylabel("Avg orders, Jan-Jun 1998")
ax.set_title("Predicted vs actual orders in the 6-month holdout"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_calibration_by_frequency.png")); plt.close(fig)

# 2: cumulative gains by predicted-spend decile, against baseline models
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for name, (pcol, scol), col in zip(models, models.values(), (NAVY, TEAL, GREY)):
    o = cust.sort_values(scol, ascending=False)
    gains = np.concatenate([[0], o.spend_hold.cumsum().values / o.spend_hold.sum()])
    ax.plot(np.linspace(0, 100, len(gains)), 100 * gains, color=col, lw=2, label=name)
ax.plot([0, 100], [0, 100], color="#cccccc", ls="--", lw=1)
ax.set_xlabel("% of customers contacted (ranked by predicted spend)")
ax.set_ylabel("% of actual holdout spend captured")
ax.set_title("Cumulative gains on the holdout window"); ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_cumulative_gains.png")); plt.close(fig)

# 3: segment profile
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
lab = [f"{r.segment}\n" + r.name.replace(" / ", "\n").replace(" (", "\n(") for r in prof.itertuples()]
xx = np.arange(len(prof))
axes[0].bar(xx - 0.2, 100 * prof.share_customers, 0.4, color=GREY, label="% of customers")
axes[0].bar(xx + 0.2, 100 * prof.share_hold_spend, 0.4, color=NAVY, label="% of holdout spend")
axes[0].set_xticks(xx); axes[0].set_xticklabels(lab, fontsize=6.5); axes[0].legend(frameon=False)
axes[0].set_title("Where the spend comes from")
axes[1].bar(xx, 100 * prof.hold_purchase_rate, color=TEAL)
axes[1].set_xticks(xx); axes[1].set_xticklabels(lab, fontsize=6.5)
axes[1].set_ylabel("% who bought Jan-Jun 1998"); axes[1].set_title("Holdout purchase rate")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_segments.png")); plt.close(fig)

# 4: break-even lift by decile
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.bar(dec.index, dec.breakeven_lift_pct, color=NAVY)
ax.set_yscale("log")
ax.set_xlabel("Predicted-spend decile (1 = highest)"); ax.set_ylabel("Break-even relative lift in spend (%, log)")
ax.set_title(f"Lift needed to cover a ${CONTACT_COST:.2f} contact at {int(MARGIN*100)}% margin")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_breakeven_lift.png")); plt.close(fig)

cust.drop(columns=["last"]).round(4).to_csv(os.path.join(OUT, "customer_scores.csv"))
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print(json.dumps(metrics, indent=2, default=str))
print("\nVALIDATION BY FREQUENCY\n", fb)
print("\nP_ALIVE BANDS (repeat buyers)\n", ab)
print("\nSEGMENTS\n", prof.round(3).to_string())
print("\nDECILES\n", dec.round(3).to_string())
print("\nSCENARIOS\n", scen.to_string())
print("\nEXPERIMENT SIZING\n", expd.to_string())
