# Card-Member Churn: Predictive Risk & Prescriptive Retention

An end-to-end analytics project that predicts credit-card member attrition and, for
**each member**, recommends the single retention action that **maximizes that member's
net ROI** — balancing churn risk, account value, save-probability, loyalty, and cost.
Built on real Kaggle `BankChurners` data with a PySpark ML pipeline, an interactive
Streamlit decision tool, and a Tableau BI dashboard.

## Live Demos
<!-- Add these links after deploying -->
- **Tableau Dashboard (executive BI view):** https://public.tableau.com/app/profile/aishita.kumar/viz/Card-MemberRetentionStrategyDashboard/Dashboard1?publish=yes
- **Streamlit App (interactive decision engine):** https://sme-commercial-risk-churn-prediction-model-mvgzoszgkjggnkkewgu.streamlit.app/

## Architecture — three layers

| Layer | Tool | Role |
|-------|------|------|
| Compute engine | PySpark + scikit-learn-grade evaluation | Prediction + per-customer decision logic |
| Interactive tool | Streamlit | Live "what-if" — move sliders, watch decisions recompute |
| BI reporting | Tableau | Executive dashboard reading the engine's exported decisions |

This mirrors how real analytics teams split work: heavy compute in Python, reporting in BI.

## Pipeline

| Stage | Script | Output |
|-------|--------|--------|
| 1. Feature engineering | `01_data_processing.py` | Cleaned, vectorized dataset |
| 2. Model training & eval | `02_model_training.py` | `model_metrics.json`, scored predictions CSV |
| 3. Executive deck | `03_generate_presentation.py` | `SME_Risk_Executive_Brief.pptx` |
| 4. Tableau export | `04_export_for_tableau.py` | Multi-scenario tidy CSV for the dashboard |
| — | `app.py` | Streamlit dashboard |

**Run order:** `01` → `02` → `03` → `04`, then `streamlit run app.py`
(place `BankChurners.csv` in `./data/`). The Streamlit app and Tableau dashboard both
read pre-computed CSVs already in `./data/`, so the app can run directly.

## Model performance (held-out 20% test set — real numbers)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.985** |
| PR-AUC | 0.919 |
| Recall (churners caught) | **90%** |
| Precision | 78% |
| F1 | 0.949 |

Class imbalance (16% churn) is handled with class weights, which lifted recall from
~65% to **90%**. Top predictor by feature importance: **transaction count / velocity**
(`Total_Trans_Ct`).

## The prescriptive engine — one clean chain of logic

For every member, the engine computes **net ROI under each channel** and picks the best:
Net ROI(action) = Value_At_Stake × Save_Prob × Channel_Effectiveness(action) − Cost(action)
recommended action = argmax over {Email, Rep Call, Senior Call}

Each factor is grounded in a real column, with a distinct job:

| Concept | Derived from | Purpose |
|---------|--------------|---------|
| **Churn risk** | model P(churn) from behavior | ranking factor (weighted) |
| **Account value** | real spend × issuer margin × real tenure × product breadth | ranking factor — kept **independent of risk** so it doesn't double-count |
| **Value at stake** | account value × churn risk | economics: what we lose if they churn |
| **Save-probability** | **real behavior only**: reachability (`Months_Inactive`) + responsiveness (`Total_Ct_Chng_Q4_Q1`) | money-waste guard — a disengaged member scores low, so the ROI math declines to spend |
| **Sleeping dogs** | high risk **AND** already dormant → suppressed | the true "lost cause" filter — identified by DISENGAGEMENT, not by a high risk score |
| **Loyalty lever** | real `Months_on_book` (tunable: protect loyal ↔ rescue new) | strategic prioritization |
| **Cost-awareness** | per-channel cost vs. expected value saved | never spends more than the account is worth |
| **Segment** | real `Card_Category` (Blue/Silver/Gold/Platinum) | reporting |

**Key design decision — high-risk members are NOT written off.** A reachable member at
90%+ risk is a genuine save opportunity and gets a call. We only suppress the truly
disengaged, because the signal for "hopeless" is *behavior*, not the risk number.

The **only** non-observed inputs are the per-channel effectiveness multipliers
(Email 50% / Rep 80% / Senior 100% of base save-rate) — transparent, tunable assumptions
a production system would learn from A/B / uplift tests.

## How the optimizer picks the Risk/Value weight

Risk and value are kept as two **independent** scores. The optimizer sweeps the weighting
0–100% and, for each, measures the **real** captured Net ROI and the **real** number of
at-risk members reached, then picks the weight that **balances both**. The typical optimum
is a broad plateau around **45–50% risk** — a genuine strategic balance, not a lopsided
dollar grab. (The Tableau version compares three revenue scenarios via a single toggle.)


## Limitations & Next Steps

1. **Save-probability and channel effectiveness are assumptions.** Anchored to real behavior,
   but not learned from outcomes. The key production upgrade is an **uplift model** trained on
   historical intervention results. The pipeline is built so this is a drop-in replacement.
2. **CLV is a proxy** (interchange margin on spend; tenure from months-on-book). A real issuer
   would use actual margin data and a survival model.
3. **The model is static** — trained and scored once. Production would retrain on a schedule
   and monitor for drift.
4. **Fairness check covers age only.** A bank would evaluate across protected attributes more
   broadly with formal fairness metrics.
5. **Small premium segments.** Gold/Platinum tiers have few members; tier-level conclusions
   there are low-confidence.

## Tech stack
PySpark · scikit-learn · pandas · Streamlit · Altair · Tableau · python-pptx

## Data
Kaggle *Credit Card Customers* (`BankChurners.csv`). Consumer credit-card data;
the SME/business framing from an earlier version has been corrected to card-member churn.
