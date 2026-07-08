"""
PHASE 4 — EXPORT FOR TABLEAU (multi-scenario, tidy)
---------------------------------------------------
Runs the SAME decision engine as the Streamlit app across three revenue scenarios
(low / base / high interchange margin) and writes ONE tidy CSV with a `Scenario`
column. In Tableau you drop `Scenario` on a filter/parameter to toggle between them.

Each row = one member under one scenario, with the engine's decision already
computed (recommended action, Net ROI, whether they're in the target list, etc.).
Nothing here is random; every field traces to a real column.
"""

import numpy as np
import pandas as pd

SRC = "data/SME_Churn_Predictions_Prioritized.csv"
OUT = "data/Tableau_Retention_Scenarios.csv"

# Fixed operating assumptions (match the Streamlit defaults)
COST_VP, COST_REP, COST_EMAIL = 40.0, 15.0, 0.10
LOYALTY = 0.0
CAPACITY = 100
EFFECTIVENESS = {"Automated Email": 0.5, "Rep Call + 5% Waiver": 0.8, "Senior Call + 10% Waiver": 1.0}

SCENARIOS = {
    "Low Margin (1.5%)":  0.015,
    "Base Margin (2.5%)": 0.025,
    "High Margin (3.5%)": 0.035,
}


def score_scenario(df_in, margin, scenario_name):
    df = df_in.copy()
    risk = df["Churn_Risk_Score"]

    # Account value (independent of risk; tenure from real Months_on_book)
    df["Expected_Retention_Years"] = np.clip(df["Months_on_book"] / 12, 1.0, 6.0)
    df["Total_Account_Value"] = (df["Total_Trans_Amt"] * margin
                                 * df["Expected_Retention_Years"]
                                 * (1.0 + (df["Total_Relationship_Count"] - 1) / 5 * 0.5))

    # Save-probability from real behavior (reachability + responsiveness)
    reachable = np.clip(1 - df["Months_Inactive_12_mon"] / 6, 0.1, 1.0)
    responsive = np.clip(df["Total_Ct_Chng_Q4_Q1"], 0.1, 1.0)
    df["Save_Prob"] = np.clip(0.10 + 0.70 * (0.5 * reachable + 0.5 * responsive), 0.05, 0.85)

    # Sleeping dogs
    df["Is_Sleeping_Dog"] = (risk > 0.85) & (df["Months_Inactive_12_mon"] >= 4) & (df["Total_Ct_Chng_Q4_Q1"] < 0.4)

    # Value at stake + per-customer argmax action
    df["Value_At_Stake"] = df["Total_Account_Value"] * risk
    action_cost = {
        "Automated Email": pd.Series(COST_EMAIL, index=df.index),
        "Rep Call + 5% Waiver": COST_REP + (df["Total_Trans_Amt"] / 12) * 0.05,
        "Senior Call + 10% Waiver": COST_VP + (df["Total_Trans_Amt"] / 12) * 0.10,
    }
    roi = {a: df["Value_At_Stake"] * df["Save_Prob"] * e - action_cost[a] for a, e in EFFECTIVENESS.items()}
    rdf = pd.DataFrame(roi)
    best_action, best_roi = rdf.idxmax(axis=1), rdf.max(axis=1)
    best_action = best_action.where(best_roi > 0, "Automated Email")
    best_roi = best_roi.where(best_roi > 0, rdf["Automated Email"])
    best_action[df["Is_Sleeping_Dog"]] = "Flagged: Sleeping Dog"
    best_roi[df["Is_Sleeping_Dog"]] = 0.0
    df["Recommended_Action"] = best_action
    df["Net_ROI"] = best_roi

    # Loyalty weight + independent value normalization
    mob = df["Months_on_book"]
    tn = (mob - mob.min()) / (mob.max() - mob.min())
    tgt = tn if LOYALTY >= 0 else (1 - tn)
    df["Loyalty_Weight"] = 1 + abs(LOYALTY) * 0.5 * tgt
    df["Normalized_Value"] = df["Total_Account_Value"] / df["Total_Account_Value"].max()

    # Optimizer: balanced objective -> optimal risk weight for THIS scenario
    hi = (risk > 0.5).values
    net = df["Net_ROI"].values
    rows = []
    for w in range(0, 101, 2):
        s = ((risk * w / 100 + df["Normalized_Value"] * (100 - w) / 100) * df["Loyalty_Weight"]).copy()
        s[df["Is_Sleeping_Dog"]] = -1
        idx = np.argpartition(s.values, -CAPACITY)[-CAPACITY:]
        rows.append((w, float(net[idx].sum()), int(hi[idx].sum())))
    R = pd.DataFrame(rows, columns=["w", "d", "r"])
    dn = (R.d - R.d.min()) / (R.d.max() - R.d.min() + 1e-9)
    rn = (R.r - R.r.min()) / (R.r.max() - R.r.min() + 1e-9)
    R["o"] = 0.5 * dn + 0.5 * rn
    best_w = int(R.loc[R.o.idxmax(), "w"])

    # Final priority at the optimal weight; flag target list + rank
    prio = ((risk * best_w / 100 + df["Normalized_Value"] * (100 - best_w) / 100) * df["Loyalty_Weight"]).copy()
    prio[df["Is_Sleeping_Dog"]] = -1
    df["Priority_Score"] = prio
    df["Priority_Rank"] = prio.rank(ascending=False, method="first").astype(int)
    df["In_Target_List"] = df["Priority_Rank"] <= CAPACITY

    # Tidy, human-readable fields
    df["Scenario"] = scenario_name
    df["Scenario_Interchange_Pct"] = margin * 100
    df["Optimal_Risk_Weight_Pct"] = best_w
    df["Recommended_Allocation"] = f"{best_w}% Risk / {100 - best_w}% Value"
    df["Card_Tier"] = df["Card_Category"]
    df["Tenure_Band"] = pd.cut(df["Months_on_book"], [0, 24, 36, 48, 100],
                               labels=["New (<2y)", "2-3y", "3-4y", "Loyal (4y+)"]).astype(str)
    df["Churn_Risk_Pct"] = (df["Churn_Risk_Score"] * 100).round(1)
    return df


def main():
    base = pd.read_csv(SRC).reset_index(drop=True)
    base.insert(0, "Customer_ID", ["M" + str(i).zfill(5) for i in range(len(base))])

    parts = [score_scenario(base, m, name) for name, m in SCENARIOS.items()]
    full = pd.concat(parts, ignore_index=True)

    keep = [
        "Scenario", "Scenario_Interchange_Pct", "Optimal_Risk_Weight_Pct", "Recommended_Allocation",
        "Customer_ID", "Recommended_Action", "In_Target_List", "Priority_Rank",
        "Churn_Risk_Score", "Churn_Risk_Pct", "Total_Account_Value", "Value_At_Stake",
        "Net_ROI", "Save_Prob", "Is_Sleeping_Dog",
        "Card_Tier", "Tenure_Band", "Customer_Age", "Months_on_book",
        "Total_Relationship_Count", "Total_Trans_Amt", "Total_Trans_Ct", "Months_Inactive_12_mon",
    ]
    out = full[keep].copy()
    for c in ["Total_Account_Value", "Value_At_Stake", "Net_ROI", "Save_Prob"]:
        out[c] = out[c].round(2)
    out.to_csv(OUT, index=False)

    print(f"Wrote {OUT}: {len(out):,} rows ({len(SCENARIOS)} scenarios x {len(base):,} members)")
    for name in SCENARIOS:
        s = out[out.Scenario == name]
        tl = s[s.In_Target_List]
        print(f"  {name:20s} -> optimal {s.Optimal_Risk_Weight_Pct.iloc[0]}% risk | "
              f"target-list ROI ${tl.Net_ROI.sum():,.0f} | "
              f"actions: {tl.Recommended_Action.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
