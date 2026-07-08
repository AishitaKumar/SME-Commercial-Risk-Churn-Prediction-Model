"""
CARD-MEMBER RETENTION — PRESCRIPTIVE ANALYTICS DASHBOARD (v4)
============================================================
For each member the engine picks the intervention that MAXIMIZES that member's
own net ROI, given real churn risk, real account value, a real-behavior
save-probability, a loyalty/tenure lever, and intervention costs.

DESIGN (one clean chain of logic, each piece anchored to a real column):
  * Rank      : weighted blend of two INDEPENDENT scores — churn risk & account value.
                The optimizer chooses the weight to balance dollars protected AND
                at-risk coverage. Risk is a full, honest factor here.
  * Save-prob : how likely an action actually retains the member, driven ONLY by REAL
                behavior — reachability (Months_Inactive) + responsiveness
                (Total_Ct_Chng_Q4_Q1). A disengaged member scores low, so the ROI
                math itself declines to spend on them (money-waste guard).
  * Lost cause: "sleeping dogs" — high risk AND already gone dark — are suppressed.
                We identify who's hopeless by DISENGAGEMENT, not by a high risk score,
                so reachable high-risk members are still worth calling.
  * Action    : per-customer argmax over channels; never spends more than the value
                at stake justifies (cost-aware).

The only non-observed inputs are the per-channel effectiveness multipliers, which
are transparent, tunable assumptions (in production: learned from A/B / uplift tests).
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Card Retention Optimizer", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    .stApp { background-color:#FFF2E8; color:#2D3748; font-family:'Inter','Helvetica Neue',sans-serif; }
    .main-title { color:#1A202C; font-weight:800; font-size:2.2rem; padding-bottom:.5rem;
        border-bottom:2px solid #FED7AA; margin-bottom:2rem; }
    .section-header { color:#C2410C; font-weight:600; font-size:1.4rem; margin:1rem 0;
        text-transform:uppercase; letter-spacing:1px; }
    div[data-testid="stMetric"] { background:#FFF9F5; border:1px solid #FFEDD5;
        border-left:4px solid #EA580C; padding:15px 20px; border-radius:8px;
        box-shadow:0 4px 10px rgba(0,0,0,.04); }
    div[data-testid="stMetric"] label { color:#718096 !important; font-weight:500; }
    div[data-testid="stMetric"] div { color:#1A202C !important; }
    [data-testid="stSidebar"] { background:#FFF5EE; border-right:1px solid #FED7AA; }
    .stDataFrame { box-shadow:0 4px 10px rgba(0,0,0,.03); border-radius:8px;
        background:#FFF9F5; border:1px solid #FFEDD5; }
    div.stSuccess { background:#ECFDF5 !important; border:1px solid #A7F3D0 !important; color:#065F46 !important; }
    div.stError { background:#FEF2F2 !important; border:1px solid #FECACA !important; color:#991B1B !important; }
    div.stWarning { background:#FFFBEB !important; border:1px solid #FDE68A !important; color:#92400E !important; }
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.markdown("<h3 style='color:#2D3748;font-size:1.1rem;'>"
    "<i class='fa-solid fa-file-invoice-dollar' style='margin-right:8px;color:#EA580C;'></i>"
    "Intervention Unit Costs</h3>", unsafe_allow_html=True)
cost_vp    = st.sidebar.number_input("Senior Retention Call Cost ($)", value=40.0, step=5.0)
cost_rep   = st.sidebar.number_input("Retention Rep Call Cost ($)",    value=15.0, step=5.0)
cost_email = st.sidebar.number_input("Automated Email Cost ($)",       value=0.10, step=0.05)

st.sidebar.markdown("<br><h3 style='color:#2D3748;font-size:1.1rem;'>"
    "<i class='fa-solid fa-percent' style='margin-right:8px;color:#EA580C;'></i>"
    "Revenue Assumptions</h3>", unsafe_allow_html=True)
interchange = st.sidebar.slider("Issuer margin on spend (interchange %)", 1.0, 4.0, 2.5, 0.25) / 100.0

st.sidebar.markdown("<br><h3 style='color:#2D3748;font-size:1.1rem;'>"
    "<i class='fa-solid fa-heart' style='margin-right:8px;color:#EA580C;'></i>"
    "Loyalty Lever</h3>", unsafe_allow_html=True)
loyalty = st.sidebar.slider("Rescue new  \u2190\u2192  Protect loyal", -1.0, 1.0, 0.0, 0.1,
                            help="Left: prioritize newer members (early churn hurts more). "
                                 "Right: prioritize long-tenure members. Center: neutral.")

st.sidebar.markdown("<br><h3 style='color:#2D3748;font-size:1.1rem;'>"
    "<i class='fa-solid fa-sliders' style='margin-right:8px;color:#EA580C;'></i>"
    "Operational Constraints</h3>", unsafe_allow_html=True)
call_capacity = st.sidebar.slider("Retention Team Capacity (Interventions/Month)", 10, 500, 100, 10)

# Per-channel effectiveness (transparent, tunable assumptions)
EFFECTIVENESS = {"Automated Email": 0.5, "Rep Call + 5% Waiver": 0.8, "Senior Call + 10% Waiver": 1.0}


@st.cache_data
def load_data(vp_c, rep_c, email_c, margin, loyalty_lever):
    df = pd.read_csv("data/SME_Churn_Predictions_Prioritized.csv")
    risk = df["Churn_Risk_Score"]

    # ---- REAL account value (INDEPENDENT of churn risk) ----
    # Expected remaining tenure is proxied from REALIZED tenure (Months_on_book),
    # so "value" is a pure worth measure and does not double-count risk.
    df["Annual_Margin"] = df["Total_Trans_Amt"] * margin
    df["Expected_Retention_Years"] = np.clip(df["Months_on_book"] / 12, 1.0, 6.0)
    df["CLV"] = df["Annual_Margin"] * df["Expected_Retention_Years"]
    df["Relationship_Multiplier"] = 1.0 + (df["Total_Relationship_Count"] - 1) / 5 * 0.5
    df["Total_Account_Value"] = df["CLV"] * df["Relationship_Multiplier"]
    df["Segment"] = df["Card_Category"]

    # ---- SAVE-PROBABILITY from REAL behavior only (no assumed risk band) ----
    # How likely an intervention actually retains this member:
    #   reachable   = still active recently (low Months_Inactive)
    #   responsive  = recent transaction frequency holding up (Total_Ct_Chng_Q4_Q1)
    # A disengaged member scores low here, so the ROI math itself declines to spend
    # on them. This is the money-waste guard, grounded in behavior rather than a curve.
    reachable = np.clip(1 - df["Months_Inactive_12_mon"] / 6, 0.1, 1.0)
    responsive = np.clip(df["Total_Ct_Chng_Q4_Q1"], 0.1, 1.0)
    df["Save_Prob"] = np.clip(0.10 + 0.70 * (0.5 * reachable + 0.5 * responsive), 0.05, 0.85)

    # ---- Sleeping dogs: high risk AND already gone dark = the true lost causes ----
    # Identified by DISENGAGEMENT, not by a high risk score, so reachable high-risk
    # members are NOT written off.
    df["Is_Sleeping_Dog"] = (risk > 0.85) & (df["Months_Inactive_12_mon"] >= 4) & (df["Total_Ct_Chng_Q4_Q1"] < 0.4)

    # ---- Value at stake = what we lose if they churn ----
    df["Value_At_Stake"] = df["Total_Account_Value"] * risk

    # ---- PER-CUSTOMER BEST ACTION (argmax net ROI over channels) ----
    action_cost = {
        "Automated Email": pd.Series(email_c, index=df.index),
        "Rep Call + 5% Waiver": rep_c + (df["Total_Trans_Amt"] / 12) * 0.05,
        "Senior Call + 10% Waiver": vp_c + (df["Total_Trans_Amt"] / 12) * 0.10,
    }
    roi = {a: df["Value_At_Stake"] * df["Save_Prob"] * eff - action_cost[a]
           for a, eff in EFFECTIVENESS.items()}
    roi_df = pd.DataFrame(roi)
    best_action = roi_df.idxmax(axis=1)
    best_roi = roi_df.max(axis=1)
    email_roi = roi_df["Automated Email"]
    best_action = best_action.where(best_roi > 0, "Automated Email")
    best_roi = best_roi.where(best_roi > 0, email_roi)
    best_action[df["Is_Sleeping_Dog"]] = "Flagged: Sleeping Dog"
    best_roi[df["Is_Sleeping_Dog"]] = 0.0
    df["Intervention_Type"] = best_action
    df["Net_ROI"] = best_roi
    df["Expected_Value_Saved"] = df["Value_At_Stake"] * df["Save_Prob"]
    df.loc[df["Is_Sleeping_Dog"], "Expected_Value_Saved"] = 0.0

    # ---- Loyalty weight (strategic prioritization lever; real tenure, risk-independent) ----
    mob = df["Months_on_book"]
    tenure_norm = (mob - mob.min()) / (mob.max() - mob.min())
    target = tenure_norm if loyalty_lever >= 0 else (1 - tenure_norm)
    df["Loyalty_Weight"] = 1 + abs(loyalty_lever) * 0.5 * target

    df["Normalized_Value"] = df["Total_Account_Value"] / df["Total_Account_Value"].max()
    return df


df = load_data(cost_vp, cost_rep, cost_email, interchange, loyalty)

# ================= HEADER =================
st.markdown("<div class='main-title'><i class='fa-solid fa-building-columns' "
    "style='color:#EA580C;margin-right:12px;'></i>Card-Member Retention "
    "Analytics & Strategy Dashboard</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#4A5568;font-size:1.1rem;'>For every member, the engine picks the "
    "<strong style='color:#1A202C;'>single action that maximizes that member's net ROI</strong> "
    "\u2014 balancing predictive churn risk, account value, save-probability, loyalty, and cost.</p>",
    unsafe_allow_html=True)


def priority_scores(data, risk_w):
    # Clean weighted blend of two INDEPENDENT scores: churn risk and account value.
    # Loyalty (tenure-based, risk-independent) is a mild tilt. Sleeping dogs excluded.
    val_w = 100 - risk_w
    s = ((data["Churn_Risk_Score"] * risk_w / 100)
         + (data["Normalized_Value"] * val_w / 100)) * data["Loyalty_Weight"]
    s = s.copy()
    s[data["Is_Sleeping_Dog"]] = -1
    return s


def find_optimal_split(data, capacity):
    roi = data["Net_ROI"].values
    high_risk = (data["Churn_Risk_Score"] > 0.5).values
    rows = []
    for w in range(0, 101, 2):
        s = priority_scores(data, w).values
        idx = (np.argpartition(s, -capacity)[-capacity:] if len(s) > capacity else np.arange(len(s)))
        rows.append({"Risk Weight (%)": w,
                     "Captured Net ROI ($)": float(np.sum(roi[idx])),
                     "At-Risk Members Reached": int(np.sum(high_risk[idx]))})
    curve = pd.DataFrame(rows)
    # Balanced objective: chosen weight maximizes BOTH money protected AND coverage of
    # at-risk members (each min-max normalized, then averaged).
    d = curve["Captured Net ROI ($)"]
    r = curve["At-Risk Members Reached"]
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-9)
    curve["_obj"] = 0.5 * d_norm + 0.5 * r_norm
    best_w = int(curve.loc[curve["_obj"].idxmax(), "Risk Weight (%)"])
    return best_w, curve.drop(columns="_obj")


optimal_risk_w, tradeoff_df = find_optimal_split(df, call_capacity)
optimal_val_w = 100 - optimal_risk_w
pop_mean_age = df["Customer_Age"].mean()


def donut(sub):
    a = sub["Intervention_Type"].value_counts().reset_index()
    a.columns = ["Intervention", "Count"]
    return (alt.Chart(a).mark_arc(innerRadius=50).encode(
        theta=alt.Theta("Count:Q"),
        color=alt.Color("Intervention:N", scale=alt.Scale(
            domain=["Senior Call + 10% Waiver", "Rep Call + 5% Waiver",
                    "Automated Email", "Flagged: Sleeping Dog"],
            range=["#4F46E5", "#F59E0B", "#10B981", "#EF4444"]),
            legend=alt.Legend(title=None, orient="bottom", labelColor="#4A5568")),
        tooltip=["Intervention", "Count"],
    ).properties(height=280).configure_view(strokeOpacity=0).configure(background="transparent"))


# ================= STRATEGY COLUMNS =================
c1, c2 = st.columns(2)
with c1:
    st.markdown("<div class='section-header'><i class='fa-solid fa-microchip' "
        "style='margin-right:10px;'></i>Data-Driven Optimized Strategy</div>", unsafe_allow_html=True)
    st.success(f"**Recommended Allocation:** {optimal_risk_w}% Risk / {optimal_val_w}% Value")
    d_opt = df.copy()
    d_opt["Live_Priority"] = priority_scores(d_opt, optimal_risk_w)
    top_opt = d_opt.nlargest(call_capacity, "Live_Priority")
    opt_rev = top_opt["Net_ROI"].sum(); opt_age = top_opt["Customer_Age"].mean()

with c2:
    st.markdown("<div class='section-header' style='color:#718096;'>"
        "<i class='fa-solid fa-user-pen' style='margin-right:10px;'></i>Manual Override Strategy</div>",
        unsafe_allow_html=True)
    manual_risk_w = st.slider("Adjust Strategy Weighting (Risk %)", 0, 100, 50, 5)
    d_man = df.copy()
    d_man["Live_Priority"] = priority_scores(d_man, manual_risk_w)
    top_man = d_man.nlargest(call_capacity, "Live_Priority")
    man_rev = top_man["Net_ROI"].sum(); man_age = top_man["Customer_Age"].mean()

m1, m2 = st.columns(2)
with m1:
    st.metric(f"Net Profit Protected (Top {call_capacity})", f"${opt_rev:,.2f}")
with m2:
    st.metric(f"Net Profit Protected (Top {call_capacity})", f"${man_rev:,.2f}",
              delta=f"${man_rev - opt_rev:,.2f} vs Optimum")

f1c, f2c = st.columns(2)
for colc, age in [(f1c, opt_age), (f2c, man_age)]:
    with colc:
        if abs(pop_mean_age - age) > 3.0:
            st.error(f"**Compliance Alert:** Targeted avg age ({age:.1f}) deviates from population ({pop_mean_age:.1f}).")
        else:
            st.success(f"**Fairness Check Passed:** Target Avg: {age:.1f} yrs | Pop Avg: {pop_mean_age:.1f} yrs")

st.divider()

cols = ["Intervention_Type", "Churn_Risk_Score", "Total_Account_Value", "Net_ROI", "Segment"]
t1, t2 = st.columns(2)
with t1:
    st.dataframe(top_opt[cols], use_container_width=True, hide_index=True)
with t2:
    st.dataframe(top_man[cols], use_container_width=True, hide_index=True)

st.divider()

ch1, ch2 = st.columns(2)
for colc, sub in [(ch1, top_opt), (ch2, top_man)]:
    with colc:
        st.markdown(f"<p style='color:#4A5568;font-weight:600;text-align:center;'>Executed Campaign (Top {call_capacity})</p>", unsafe_allow_html=True)
        st.altair_chart(donut(sub), use_container_width=True, theme=None)

st.divider()

# ================= WHO ARE WE TARGETING =================
st.markdown("<div class='section-header'><i class='fa-solid fa-people-group' "
    "style='margin-right:10px;'></i>Who Are We Targeting?</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#718096;margin-bottom:16px;'>Profile of the optimized target list "
    "\u2014 so the segment story is visible, not hidden in the numbers.</p>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg Churn Risk", f"{top_opt['Churn_Risk_Score'].mean():.0%}")
k2.metric("Avg Tenure", f"{top_opt['Months_on_book'].mean():.0f} mo")
k3.metric("Avg Account Value", f"${top_opt['Total_Account_Value'].mean():,.0f}")
n_calls = int(top_opt['Intervention_Type'].str.contains('Call').sum())
k4.metric("High-Touch Calls", f"{n_calls} / {call_capacity}")

seg1, seg2 = st.columns(2)
with seg1:
    seg = top_opt["Segment"].value_counts().reset_index()
    seg.columns = ["Segment", "Count"]
    st.markdown("<p style='color:#4A5568;font-weight:600;'>By Card Tier</p>", unsafe_allow_html=True)
    st.altair_chart(alt.Chart(seg).mark_bar(color="#EA580C").encode(
        x=alt.X("Count:Q"), y=alt.Y("Segment:N", sort="-x"), tooltip=["Segment", "Count"]
    ).properties(height=180).configure(background="transparent").configure_view(strokeOpacity=0),
        use_container_width=True, theme=None)
with seg2:
    tb = top_opt.copy()
    tb["Tenure Band"] = pd.cut(tb["Months_on_book"], [0, 24, 36, 48, 100],
                               labels=["New (<2y)", "2\u20133y", "3\u20134y", "Loyal (4y+)"])
    tband = tb["Tenure Band"].value_counts().reset_index()
    tband.columns = ["Tenure Band", "Count"]
    st.markdown("<p style='color:#4A5568;font-weight:600;'>By Tenure</p>", unsafe_allow_html=True)
    st.altair_chart(alt.Chart(tband).mark_bar(color="#4F46E5").encode(
        x=alt.X("Count:Q"), y=alt.Y("Tenure Band:N", sort="-x"), tooltip=["Tenure Band", "Count"]
    ).properties(height=180).configure(background="transparent").configure_view(strokeOpacity=0),
        use_container_width=True, theme=None)

st.divider()

# ================= TRADEOFF CURVE (two REAL lines) =================
st.markdown("<div class='section-header'><i class='fa-solid fa-chart-line' "
    "style='margin-right:10px;'></i>Risk vs. Value Tradeoff (Actual Outcomes)</div>",
    unsafe_allow_html=True)
st.markdown("<p style='color:#718096;margin-bottom:20px;'>Both lines are computed directly from "
    "the data as the Risk/Value weighting sweeps. <strong>Net ROI captured</strong> is the money "
    "protected; <strong>At-risk members reached</strong> is how many likely-churners the same "
    "capacity touches. The optimum balances both.</p>", unsafe_allow_html=True)

base = alt.Chart(tradeoff_df).encode(
    x=alt.X("Risk Weight (%):Q", axis=alt.Axis(gridColor="#FFEDD5", labelColor="#718096")))
roi_line = base.mark_line(strokeWidth=3, color="#EA580C").encode(
    y=alt.Y("Captured Net ROI ($):Q", axis=alt.Axis(title="Captured Net ROI ($)",
            gridColor="#FFEDD5", labelColor="#EA580C", titleColor="#EA580C")),
    tooltip=["Risk Weight (%)", "Captured Net ROI ($)"])
reach_line = base.mark_line(strokeWidth=3, color="#3B82F6", strokeDash=[4, 3]).encode(
    y=alt.Y("At-Risk Members Reached:Q", axis=alt.Axis(title="At-Risk Members Reached",
            labelColor="#3B82F6", titleColor="#3B82F6")),
    tooltip=["Risk Weight (%)", "At-Risk Members Reached"])
vline = alt.Chart(pd.DataFrame({"x": [optimal_risk_w]})).mark_rule(
    strokeDash=[5, 5], strokeWidth=2, color="#1A202C").encode(x="x:Q")

st.altair_chart(alt.layer(roi_line, reach_line, vline).resolve_scale(y="independent")
                .properties(height=400).configure(background="transparent")
                .configure_view(strokeOpacity=0), use_container_width=True, theme=None)
st.caption(f"**Empirical Optimum:** the balanced objective peaks at **{optimal_risk_w}%** Risk / "
           f"**{optimal_val_w}%** Value for the current capacity and cost settings. "
           f"Channel effectiveness assumptions: Email 50% / Rep 80% / Senior 100% of base save-rate "
           f"(tunable; learned from A/B tests in production).")
st.divider()