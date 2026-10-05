# Customer segmentation, CLV and targeting economics on CDNOW purchase data

This project asks a practical marketing question: given a base of customers and a limited contact budget, who is worth spending on, and how would you prove the spend worked? It uses real purchase history from CDNOW, an online music retailer, and keeps a strict line between what was measured and what was assumed.

The code was written with AI assistance. The design, the validation choices and the reading of the results are mine.

## Data

The CDNOW file has 69,659 line items from 23,570 customers between January 1997 and June 1998. I dropped 80 zero-value lines, collapsed same-day lines into single orders (67,511 orders), and ended up with 23,502 customers. Every customer made a first purchase in 1997 (January 1 to November 23).

Models are fit on orders through December 31, 1997. They are then checked against what the same customers did in the next 181 days (January to June 1998). No holdout data touches any fit or any tuning choice.

## What the work does

1. Segments customers on recency, frequency and spend (log-scaled, k-means, k=5).
2. Forecasts each customer's next six months with a BG/NBD model for order counts and a Gamma-Gamma model for order value.
3. Checks those forecasts against the holdout, and against two simpler alternatives.
4. Turns the forecasts into a targeting rule: the lift a campaign has to deliver to cover its cost, by customer decile.
5. Sizes the test you would run to confirm the lift is real.

## Results

**Aggregate forecast.** The model predicted 12,429 orders and $494K of spend for the holdout. Actual was 12,265 orders and $476K, so +1.3% on orders and +3.8% on spend.

**Ranking customers.** Out of the 23,502 customers, the top 10% by predicted spend produced 56.6% of actual holdout spend, and the top 20% produced 72.4%.

**Against simpler models.** The BG/NBD approach did not clearly win on ranking. A gradient-boosting model on the same RFM inputs, scored out-of-fold, landed in the same place.

| Model | AUC (bought at all) | Share of spend in top 10% | Spearman vs actual spend |
|---|---|---|---|
| BG/NBD + Gamma-Gamma | 0.799 | 56.6% | 0.427 |
| Gradient boosting | 0.806 | 56.6% | 0.444 |
| Past rate carried forward | 0.793 | 53.2% | 0.484 |

The Spearman column favours the naive rule, which I would not read much into: 77% of customers spent nothing in the holdout, so the metric is mostly measuring ties. The reason to prefer BG/NBD here is that it has four parameters, produces a usable aggregate forecast, and can be explained to a client in a sentence. It is not that it ranks better.

**Segments.** Silhouette scores for k=4 to 7 only range from 0.41 to 0.45, so the statistics do not pick k. I chose five because a marketing team can act on five. Re-fitting with a different random seed gave nearly identical segments (adjusted Rand index 0.97).

| Segment | Customers | Share of holdout spend | Bought Jan-Jun 1998 |
|---|---|---|---|
| S1 Heavy repeat / active | 6.9% | 47.9% | 79.5% |
| S2 Mid repeat / active | 9.1% | 14.1% | 48.2% |
| S3 Mid repeat / cooling | 14.8% | 20.9% | 39.1% |
| S4 One-time / lapsed (higher spend) | 30.0% | 11.1% | 13.7% |
| S5 One-time / lapsed (lower spend) | 39.3% | 6.0% | 8.0% |

## Where the model falls short

**It writes off quiet repeat buyers too early.** Among customers with at least one repeat order, those the model rated under 0.2 probability of still being active placed 0.40 orders on average in the holdout. The model had predicted 0.135. About 19% of them bought at all, against 8% for the lapsed one-time buyers. This is probably why the lowest-ranked decile behaves oddly (12.8% of it bought, against 8% in the deciles above): 64% of that decile has at least one repeat order. I did not test whether every one of them sits in the low P(alive) band. In practice, a low P(alive) should not be used as a hard "churned" flag, and this group is a reasonable win-back test.

**It misses at both ends of frequency.** It under-predicts customers with one or two repeat orders (0.28 predicted vs 0.39 actual for one) and over-predicts heavy buyers (4.23 vs 3.81 for seven or more).

**P(alive) is mechanically 1.0 for one-time buyers.** Under BG/NBD a customer cannot drop out before a second purchase, so the metric says nothing about whether a one-time buyer is gone. I do not use it for them.

**The data is old and narrow.** One retailer, a 1997 to 1998 acquisition cohort, no channel or campaign fields.

## Targeting economics

Contact cost ($0.60) and contribution margin (35%) are my assumptions, set to plausible values. They are not drawn from the data. Everything below that depends on them is a scenario.

The useful output is the break-even lift, the relative increase in six-month spend a campaign must produce for a contact to pay back, using predicted spend per decile:

| Decile (1 = highest predicted spend) | Break-even lift |
|---|---|
| 1 | 1.3% |
| 2 | 5.3% |
| 3 | 12.5% |
| 4 | 27.6% |
| 5 to 9 | 35% to 40% |
| 10 | 54% |

That gives a rule that needs no hindsight: contact a decile only if you believe the campaign can beat its break-even lift. Deciles 1 and 2 clear a low bar. From decile 4 down, the campaign has to lift spend by 28% or more.

For scale, I applied assumed lifts to actual holdout spend. At $0.60 per contact and a 5% lift, contacting everyone loses about $5.8K over the window, while targeting the best 13.8% by predicted spend nets about $3.4K. The targeting depth in that table is picked with hindsight on the holdout, so it shows the shape of the trade-off and is not a forecast. Full grid in `outputs/targeting_scenarios.csv`.

## Testing the lift

Because the lift is assumed, the next step is a holdout experiment. Required sample per arm at 80% power and 5% significance, for a 20% relative lift in purchase rate:

| Segment | Baseline purchase rate | Per arm (purchase rate) | Per arm (spend) |
|---|---|---|---|
| S1 | 79.5% | 61 | 1,225 |
| S2 | 48.2% | 420 | 1,165 |
| S3 | 39.1% | 628 | 1,873 |
| S4 | 13.7% | 2,683 | 6,945 |
| S5 | 8.0% | 4,909 | 10,514 |

Testing on purchase rate is far cheaper than testing on spend, because spend is so skewed. S5 cannot support a two-arm test on its own: it needs 9,818 customers and the segment has 9,233. For a 10% lift in purchase rate, only S1 is large enough to test on its own; every other segment would need to be pooled.

## What this is not

There is no media, channel or campaign data in this set, so this is not attribution or a marketing mix model, and nothing here measures incrementality. The lift figures are assumptions until a test is run.

## Files

- `clv_pipeline.py` runs everything and writes to `outputs/`
- `outputs/metrics.json`, `model_comparison.csv`, `segment_profile.csv`, `decile_table.csv`, `targeting_scenarios.csv`, `experiment_sizing.csv`, `validation_by_frequency.csv`, `validation_by_p_alive_repeat_buyers.csv`, `customer_scores.csv`
- `outputs/fig1` to `fig4`: calibration, cumulative gains, segments, break-even lift

Dependencies: pandas, numpy, scikit-learn, lifetimes, statsmodels, scipy, matplotlib. The data ships with the `lifetimes` package.
