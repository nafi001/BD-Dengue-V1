"""
Bangladesh Dengue Early-Warning & Multi-Horizon Forecasting Dashboard
Includes: Interactive Division Map, 1-14d Probabilistic Forecast Chart,
Division & Year Filters, Descriptive Surveillance Graphs, CSV Upload & Manual QA Sign-off.
Note: Explainable AI (SHAP) outputs are strictly reserved for research reporting.
"""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_quality import audit_all_divisions, REQUIRED_RAW_COLS
from src.inference import (
    DIVISIONS,
    forecast_all_divisions_and_national,
    compute_weekly_outbreak_status,
)
from src.prospective_logger import (
    append_prospective_forecasts,
    append_qa_signoff,
    score_logged_forecasts,
)

APP_ROOT = Path(__file__).resolve().parent
ART_DIR = APP_ROOT / "artifacts"
LOGS_DIR = APP_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Accurate centroids for Bangladesh's 8 administrative divisions
DIV_COORDS = {
    "dhaka":      (23.8103, 90.4125),
    "chattogram": (22.3569, 91.7832),
    "khulna":     (22.8456, 89.5403),
    "rajshahi":   (24.3745, 88.6042),
    "rangpur":    (25.7439, 89.2752),
    "barisal":    (22.7010, 90.3535),
    "sylhet":     (24.8949, 91.8687),
    "mymensingh": (24.7471, 90.4203),
}

ALERT_COLOR_MAP = {
    "NORMAL": "#2CA02C",
    "WATCH (Upper 90% > P90)": "#F1C40F",
    "ALERT (> P75)": "#FF7F0E",
    "HIGH ALERT (> P90)": "#E74C3C",
    "EPIDEMIC (> Mean+2SD)": "#8E0000",
}

st.set_page_config(
    page_title="Bangladesh Dengue Early-Warning System",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_packaged_assets():
    with open(ART_DIR / "production_metadata.json", "r") as fp:
        meta = json.load(fp)
    hist_df = pd.read_parquet(ART_DIR / "full_history_all_divisions.parquet")
    hist_df["Date"] = pd.to_datetime(hist_df["Date"])
    hist_df["ADM1_NAME"] = hist_df["ADM1_NAME"].str.lower().str.strip()
    thresh_df = pd.read_csv(ART_DIR / "weekly_endemic_thresholds.csv")
    bench_df = pd.read_csv(ART_DIR / "benchmark_holdout_summary.csv")
    return meta, hist_df, thresh_df, bench_df


@st.cache_data(show_spinner="Running frozen pooled_lgb_gjr_garch 1–14d forecast engine...")
def run_cached_inference(cache_key: str, _hist_df: pd.DataFrame):
    return forecast_all_divisions_and_national(_hist_df, ART_DIR)


meta, base_hist_df, thresh_df, bench_df = load_packaged_assets()

if "active_hist_df" not in st.session_state:
    st.session_state["active_hist_df"] = base_hist_df
    st.session_state["data_source_label"] = f"Packaged DGHS Feed ({meta['data_start_date']} to {meta['latest_data_date']})"

active_hist = st.session_state["active_hist_df"].copy()
active_hist["Date"] = pd.to_datetime(active_hist["Date"])
active_hist["Year"] = active_hist["Date"].dt.year
latest_dt_str = active_hist["Date"].max().strftime("%Y-%m-%d")
available_years = sorted(active_hist["Year"].dropna().astype(int).unique().tolist())

# Run QA audit, 1-14d forecasts & outbreak status
qa_df = audit_all_divisions(active_hist, DIVISIONS, reference_date=latest_dt_str)
fc_df = run_cached_inference(f"{latest_dt_str}_{len(active_hist)}", active_hist)
ob_df = compute_weekly_outbreak_status(fc_df, thresh_df)

# ==============================================================================
# SIDEBAR FILTERS (DIVISION + YEAR)
# ==============================================================================
st.sidebar.title("🦟 BD Dengue EWS")
st.sidebar.caption("Division & National Forecasting Dashboard")
st.sidebar.markdown("---")

selected_scope = st.sidebar.selectbox(
    "Administrative Division",
    options=["national"] + DIVISIONS,
    format_func=lambda s: "National (All 8 Divisions)" if s == "national" else f"{s.capitalize()} Division",
)

selected_years = st.sidebar.multiselect(
    "Filter Year(s) (Descriptive & Map)",
    options=available_years,
    default=available_years,
)
if not selected_years:
    selected_years = available_years

history_window = st.sidebar.slider("Forecast Chart History Window (Days)", 30, 180, 90, step=15)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Engine:** `{meta['production_model']}`\n\n"
    f"**National:** Coherent Bottom-Up Sum\n\n"
    f"**Latest Data:** `{latest_dt_str}`\n\n"
    f"**Forecast Horizon:** `1–14 Days Ahead`"
)
st.sidebar.caption("Research prototype. Manual QA sign-off required prior to public bulletin release.")

# Build filtered historical series for KPIs & descriptive plots
if selected_scope == "national":
    scope_hist_all = (
        active_hist.groupby("Date", as_index=False)
        .agg({
            "dengue": "sum", "Urban_T_mean": "mean", "Urban_RH": "mean",
            "Urban_Rainfall": "mean", "Urban_NDWI": "mean", "Urban_LST_Day": "mean",
        })
        .sort_values("Date")
    )
else:
    scope_hist_all = active_hist[active_hist["ADM1_NAME"] == selected_scope].sort_values("Date").copy()

scope_hist_all["Year"] = scope_hist_all["Date"].dt.year
scope_hist_filt = scope_hist_all[scope_hist_all["Year"].isin(selected_years)].copy()

# ==============================================================================
# HEADER & 6-CARD KPI STRIP
# ==============================================================================
scope_label = "National (Bottom-Up Sum)" if selected_scope == "national" else f"{selected_scope.capitalize()} Division"
st.title("Bangladesh Dengue Forecasting & Outbreak Early-Warning Dashboard")
st.caption(f"Scope: **{scope_label}** | Years Selected: **{min(selected_years)}–{max(selected_years)}** | Feed: {st.session_state['data_source_label']}")

scope_fc = fc_df[fc_df["scope"] == selected_scope].sort_values("horizon")
scope_ob_w1 = ob_df[(ob_df["scope"] == selected_scope) & (ob_df["lead_window"].str.startswith("Week +1"))].iloc[0]
scope_ob_w2 = ob_df[(ob_df["scope"] == selected_scope) & (ob_df["lead_window"].str.startswith("Week +2"))].iloc[0]

total_cases = int(scope_hist_filt["dengue"].sum())
mean_daily = float(scope_hist_filt["dengue"].mean()) if len(scope_hist_filt) else 0.0
peak_daily = int(scope_hist_filt["dengue"].max()) if len(scope_hist_filt) else 0
n_warn = int((qa_df["status"] == "WARN").sum())
n_fail = int((qa_df["status"] == "FAIL").sum())
dq_badge = "PASS" if (n_warn == 0 and n_fail == 0) else (f"{n_fail} FAIL / {n_warn} WARN" if n_fail > 0 else f"{n_warn} WARN")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Admissions (Filtered)", f"{total_cases:,}")
k2.metric("Daily Mean / Peak", f"{mean_daily:,.1f}", delta=f"Peak: {peak_daily:,}", delta_color="off")
k3.metric("Next 7d Projected", f"{scope_ob_w1['projected_weekly_cases']:,.0f}", delta=f"P75: {scope_ob_w1['thresh_p75']:,.0f}", delta_color="off")
k4.metric("Week +1 Alert", scope_ob_w1["alert_status"].split(" ")[0])
k5.metric("Week +2 Alert", scope_ob_w2["alert_status"].split(" ")[0])
k6.metric("Data Quality", dq_badge)

# ==============================================================================
# MAIN TABS
# ==============================================================================
tab_map, tab_fc, tab_desc, tab_upload, tab_qa = st.tabs([
    "🗺️ Bangladesh Map View",
    "📈 1–14d Forecast Line Chart",
    "📊 Descriptive Analytics",
    "📤 Upload New CSV & QA",
    "🚨 Outbreak Status & Manual Release",
])

# ------------------------------------------------------------------------------
# TAB 1: INTERACTIVE BANGLADESH MAP VIEW
# ------------------------------------------------------------------------------
with tab_map:
    map_mode = st.radio(
        "Map Layer",
        options=["Live 7-Day Projected Burden & Outbreak Alert", "Historical Admissions (Selected Years)"],
        horizontal=True,
    )

    w1_divs = ob_df[(ob_df["scope"].isin(DIVISIONS)) & (ob_df["lead_window"].str.startswith("Week +1"))].copy()
    hist_div_sum = (
        active_hist[active_hist["Year"].isin(selected_years)]
        .groupby("ADM1_NAME", as_index=False)
        .agg(total_admissions=("dengue", "sum"), daily_mean=("dengue", "mean"), peak_daily=("dengue", "max"))
        .rename(columns={"ADM1_NAME": "scope"})
    )
    map_df = w1_divs.merge(hist_div_sum, on="scope", how="left")
    map_df["Division"] = map_df["scope"].str.capitalize()
    map_df["lat"] = map_df["scope"].map(lambda d: DIV_COORDS[d][0])
    map_df["lon"] = map_df["scope"].map(lambda d: DIV_COORDS[d][1])
    map_df["daily_mean"] = map_df["daily_mean"].round(1)
    map_df["total_admissions"] = map_df["total_admissions"].fillna(0).astype(int)

    col_m1, col_m2 = st.columns([1.45, 1.0])
    with col_m1:
        if map_mode.startswith("Live"):
            map_df["bubble_size"] = np.maximum(map_df["projected_weekly_cases"], 15.0)
            fig_map = px.scatter_map(
                map_df, lat="lat", lon="lon", size="bubble_size", color="alert_status",
                color_discrete_map=ALERT_COLOR_MAP, hover_name="Division",
                hover_data={
                    "projected_weekly_cases": ":.1f", "thresh_p75": ":.1f",
                    "alert_status": True, "lat": False, "lon": False, "bubble_size": False,
                },
                size_max=42, zoom=5.8, center={"lat": 23.75, "lon": 90.35},
                map_style="carto-positron",
                title=f"Projected Next-7-Day Dengue Burden & Alert Tier (Origin: {latest_dt_str})",
            )
        else:
            map_df["bubble_size"] = np.maximum(map_df["total_admissions"], 100.0)
            fig_map = px.scatter_map(
                map_df, lat="lat", lon="lon", size="bubble_size", color="total_admissions",
                color_continuous_scale="YlOrRd", hover_name="Division",
                hover_data={
                    "total_admissions": ":,", "daily_mean": ":.1f", "peak_daily": ":,",
                    "lat": False, "lon": False, "bubble_size": False,
                },
                size_max=45, zoom=5.8, center={"lat": 23.75, "lon": 90.35},
                map_style="carto-positron",
                title=f"Observed Dengue Admissions Across 8 Divisions ({min(selected_years)}–{max(selected_years)})",
            )
        fig_map.update_layout(height=540, margin={"r": 0, "t": 45, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)

    with col_m2:
        st.subheader("Division Summary Table")
        show_cols = ["Division", "projected_weekly_cases", "thresh_p75", "alert_status", "total_admissions"]
        st.dataframe(
            map_df[show_cols].sort_values("projected_weekly_cases", ascending=False).rename(columns={
                "projected_weekly_cases": "Next 7d Forecast",
                "thresh_p75": "P75 Threshold",
                "alert_status": "Week +1 Status",
                "total_admissions": f"Total ({min(selected_years)}–{max(selected_years)})",
            }),
            use_container_width=True, hide_index=True, height=340,
        )
        st.caption("Bubble sizes scale with burden. Click any legend item to filter alert tiers.")

# ------------------------------------------------------------------------------
# TAB 2: 1–14 DAY PROBABILISTIC FORECAST LINE CHART
# ------------------------------------------------------------------------------
with tab_fc:
    hist_recent = scope_hist_all.tail(history_window)
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([scope_fc["target_date"], scope_fc["target_date"][::-1]]),
        y=pd.concat([scope_fc["q975"], scope_fc["q025"][::-1]]),
        fill="toself", fillcolor="rgba(0, 114, 178, 0.14)", line=dict(color="rgba(255,255,255,0)"),
        name="95% PI", hoverinfo="skip",
    ))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([scope_fc["target_date"], scope_fc["target_date"][::-1]]),
        y=pd.concat([scope_fc["q900"], scope_fc["q100"][::-1]]),
        fill="toself", fillcolor="rgba(0, 114, 178, 0.25)", line=dict(color="rgba(255,255,255,0)"),
        name="80% PI", hoverinfo="skip",
    ))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([scope_fc["target_date"], scope_fc["target_date"][::-1]]),
        y=pd.concat([scope_fc["q750"], scope_fc["q250"][::-1]]),
        fill="toself", fillcolor="rgba(0, 114, 178, 0.40)", line=dict(color="rgba(255,255,255,0)"),
        name="50% PI", hoverinfo="skip",
    ))
    fig_fc.add_trace(go.Scatter(
        x=hist_recent["Date"].dt.strftime("%Y-%m-%d"), y=hist_recent["dengue"],
        mode="lines", line=dict(color="#1F2937", width=1.8), name="Observed Daily Admissions",
    ))
    fig_fc.add_trace(go.Scatter(
        x=scope_fc["target_date"], y=scope_fc["y_pred"],
        mode="lines+markers", line=dict(color="#0072B2", width=2.5), marker=dict(size=5),
        name="Point Forecast (pooled_lgb_gjr_garch)",
    ))
    fig_fc.update_layout(
        title=f"{scope_label} — 1–14 Day Probabilistic Dengue Forecast",
        xaxis_title="Date", yaxis_title="Daily Hospital Admissions",
        hovermode="x unified", height=480, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.subheader("Daily Forecast Table (1–14 Days Ahead)")
    st.dataframe(
        scope_fc[["target_date", "horizon", "y_pred", "q250", "q750", "q100", "q900", "q025", "q975"]].rename(columns={
            "target_date": "Target Date", "horizon": "Horizon (d)", "y_pred": "Point Forecast",
            "q250": "50% Low", "q750": "50% High", "q100": "80% Low", "q900": "80% High",
            "q025": "95% Low", "q975": "95% High",
        }),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇️ Download Full 8-Division + National 14-Day Forecast CSV",
        data=fc_df.to_csv(index=False).encode("utf-8"),
        file_name=f"bd_dengue_forecast_{latest_dt_str}.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------------------------
# TAB 3: DESCRIPTIVE ANALYTICS (FILTERED BY DIVISION & YEAR)
# ------------------------------------------------------------------------------
with tab_desc:
    st.subheader(f"Surveillance Trend & Seasonality — {scope_label} ({min(selected_years)}–{max(selected_years)})")

    # 1. Daily + 7-day rolling mean trajectory
    scope_hist_filt["ma_7d"] = scope_hist_filt["dengue"].rolling(7, min_periods=1).mean()
    fig_tr = go.Figure()
    fig_tr.add_trace(go.Scatter(
        x=scope_hist_filt["Date"], y=scope_hist_filt["dengue"],
        mode="lines", line=dict(color="rgba(31, 119, 180, 0.35)", width=1.0), name="Daily Admissions",
    ))
    fig_tr.add_trace(go.Scatter(
        x=scope_hist_filt["Date"], y=scope_hist_filt["ma_7d"],
        mode="lines", line=dict(color="#0072B2", width=2.2), name="7-Day Moving Average",
    ))
    fig_tr.update_layout(
        title="Daily Hospital Admissions & 7-Day Moving Average",
        xaxis_title="Date", yaxis_title="Admissions", height=360, template="plotly_white", hovermode="x unified",
    )
    st.plotly_chart(fig_tr, use_container_width=True)

    c_d1, c_d2 = st.columns(2)

    # 2. Monthly seasonality by Year
    with c_d1:
        scope_hist_filt["Month"] = scope_hist_filt["Date"].dt.month
        scope_hist_filt["MonthName"] = scope_hist_filt["Date"].dt.strftime("%b")
        m_agg = scope_hist_filt.groupby(["Year", "Month", "MonthName"], as_index=False)["dengue"].sum().sort_values(["Year", "Month"])
        m_agg["Year"] = m_agg["Year"].astype(str)
        fig_mo = px.bar(
            m_agg, x="MonthName", y="dengue", color="Year", barmode="group",
            category_orders={"MonthName": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]},
            title="Monthly Admissions by Year", labels={"dengue": "Monthly Total", "MonthName": "Month"},
            template="plotly_white",
        )
        fig_mo.update_layout(height=360)
        st.plotly_chart(fig_mo, use_container_width=True)

    # 3. Day-of-Week reporting pattern (Friday dip)
    with c_d2:
        dow_order = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        scope_hist_filt["Weekday"] = scope_hist_filt["Date"].dt.strftime("%a")
        dow_agg = scope_hist_filt.groupby("Weekday", as_index=False)["dengue"].mean()
        dow_agg["Color"] = np.where(dow_agg["Weekday"] == "Fri", "#D55E00", "#0072B2")
        fig_dow = px.bar(
            dow_agg, x="Weekday", y="dengue", color="Color", color_discrete_map="identity",
            category_orders={"Weekday": dow_order},
            title="Mean Admissions by Day of Week (Friday Reporting Dip Highlighted)",
            labels={"dengue": "Mean Daily Admissions"}, template="plotly_white",
        )
        fig_dow.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_dow, use_container_width=True)

    # 4. Climate / Satellite vs. Dengue dual-axis explorer
    env_var = st.selectbox(
        "Overlay Environmental / Satellite Covariate (Weekly Mean)",
        options=["Urban_T_mean", "Urban_RH", "Urban_Rainfall", "Urban_NDWI", "Urban_LST_Day"],
    )
    wk_env = (
        scope_hist_filt.set_index("Date")
        .resample("W-SAT")
        .agg({"dengue": "sum", env_var: "mean"})
        .reset_index()
    )
    fig_env = make_subplots(specs=[[{"secondary_y": True}]])
    fig_env.add_trace(go.Bar(x=wk_env["Date"], y=wk_env["dengue"], name="Weekly Dengue", marker_color="rgba(0,114,178,0.45)"), secondary_y=False)
    fig_env.add_trace(go.Scatter(x=wk_env["Date"], y=wk_env[env_var], name=env_var, line=dict(color="#D55E00", width=2.0)), secondary_y=True)
    fig_env.update_layout(title=f"Weekly Dengue Admissions vs. {env_var}", height=360, template="plotly_white", hovermode="x unified")
    fig_env.update_yaxes(title_text="Weekly Admissions", secondary_y=False)
    fig_env.update_yaxes(title_text=env_var, secondary_y=True)
    st.plotly_chart(fig_env, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: UPLOAD NEW CSV DATA & DATA QUALITY AUDIT
# ------------------------------------------------------------------------------
with tab_upload:
    st.subheader("Automated Surveillance & Satellite Data Quality Flags")
    st.dataframe(qa_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Upload New Surveillance Data (CSV) for Live Re-Inference")
    st.caption("Upload a CSV with new daily rows (columns: `ADM1_NAME`, `Date`, `dengue`, and the 13 `Urban_*` covariates). New rows merge cleanly onto historical data and refresh all charts, maps, and 1–14d forecasts.")

    # Provide a ready-to-use CSV template download
    sample_template = base_hist_df[REQUIRED_RAW_COLS].sort_values("Date").tail(16)
    st.download_button(
        "📄 Download Sample CSV Schema Template",
        data=sample_template.to_csv(index=False).encode("utf-8"),
        file_name="bd_dengue_upload_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload Updated Daily CSV", type=["csv"])
    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            missing_c = [c for c in REQUIRED_RAW_COLS if c not in new_df.columns]
            if missing_c:
                st.error(f"Uploaded CSV is missing required columns: {missing_c}")
            else:
                new_df["Date"] = pd.to_datetime(new_df["Date"])
                new_df["ADM1_NAME"] = new_df["ADM1_NAME"].str.lower().str.strip()
                merged = pd.concat([base_hist_df, new_df], ignore_index=True)
                merged = merged.sort_values(["ADM1_NAME", "Date"]).drop_duplicates(subset=["ADM1_NAME", "Date"], keep="last")
                st.session_state["active_hist_df"] = merged
                st.session_state["data_source_label"] = f"Uploaded Update ({uploaded.name}, through {merged['Date'].max().strftime('%Y-%m-%d')})"
                st.success("New data merged! Forecasts, maps, and descriptive charts refreshed.")
                st.rerun()
        except Exception as exc:
            st.error(f"Failed to ingest CSV: {exc}")

    if st.button("Reset to Packaged DGHS Feed"):
        st.session_state["active_hist_df"] = base_hist_df
        st.session_state["data_source_label"] = f"Packaged DGHS Feed ({meta['data_start_date']} to {meta['latest_data_date']})"
        st.rerun()

# ------------------------------------------------------------------------------
# TAB 5: OUTBREAK STATUS & MANUAL QA RELEASE WORKFLOW
# ------------------------------------------------------------------------------
with tab_qa:
    st.subheader("Weekly Outbreak Early-Warning Status (All 8 Divisions + National)")

    def _color_status(val):
        if "EPIDEMIC" in str(val):
            return "background-color: #FADBD8; color: #922B21; font-weight: bold;"
        if "HIGH ALERT" in str(val):
            return "background-color: #FDEBD0; color: #9C640C; font-weight: bold;"
        if "ALERT" in str(val) or "WATCH" in str(val):
            return "background-color: #FCF3CF; color: #7D6608; font-weight: bold;"
        return "background-color: #D5F5E3; color: #196F3D;"

    st.dataframe(ob_df.style.map(_color_status, subset=["alert_status"]), use_container_width=True, hide_index=True)

    with st.expander("View Frozen Hold-Out Test Benchmark Card (Sep 2025 – Sep 2026)"):
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Manual QA Release Sign-Off (Required Prior to Bulletin Distribution)")
    with st.form("qa_release_form"):
        reviewer = st.text_input("Reviewer Name / Affiliation", placeholder="e.g., M. K. Ansari, SUST")
        c1 = st.checkbox("Verified DGHS daily hospital admission feed completeness across all 8 divisions")
        c2 = st.checkbox("Checked Friday / Bangladesh public holiday reporting dips and backlog effects")
        c3 = st.checkbox("Verified 50%/80%/95% prediction intervals for physical plausibility")
        c4 = st.checkbox("Cross-checked projected Week +1 / Week +2 alerts against regional reports")
        decision = st.selectbox("Release Decision", ["APPROVE_FOR_RELEASE", "HOLD_FOR_DATA_REVIEW"])
        notes = st.text_area("QA Notes (optional)")
        submitted = st.form_submit_button("Sign Off & Log Prospective Forecast")

        if submitted:
            if not reviewer.strip():
                st.error("Please enter Reviewer Name before signing off.")
            elif decision == "APPROVE_FOR_RELEASE" and not (c1 and c2 and c3 and c4):
                st.error("All 4 QA checklist items must be checked to APPROVE_FOR_RELEASE.")
            else:
                rel_id = f"REL_{latest_dt_str}_{datetime.now(timezone(timedelta(hours=6))).strftime('%H%M%S')}"
                append_qa_signoff({
                    "release_id": rel_id, "origin_date": latest_dt_str, "reviewer": reviewer.strip(),
                    "decision": decision, "checks_passed": int(c1) + int(c2) + int(c3) + int(c4),
                    "qa_warn_divisions": n_warn, "qa_fail_divisions": n_fail, "notes": notes.strip(),
                }, LOGS_DIR / "qa_checklist_log.csv")
                if decision == "APPROVE_FOR_RELEASE":
                    n_log = append_prospective_forecasts(fc_df, LOGS_DIR / "prospective_forecasts_log.csv", release_id=rel_id)
                    st.success(f"Approved & logged {n_log} forecast rows under Release ID `{rel_id}`.")
                else:
                    st.warning(f"Logged HOLD decision under Release ID `{rel_id}`.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Recent QA Sign-Off History**")
        qa_f = LOGS_DIR / "qa_checklist_log.csv"
        if qa_f.exists() and qa_f.stat().st_size > 0:
            st.dataframe(pd.read_csv(qa_f).tail(10), use_container_width=True, hide_index=True)
        else:
            st.info("No manual QA sign-offs logged yet.")
    with col_r:
        st.markdown("**Prospective Forecast Realized Scoring**")
        sc_df = score_logged_forecasts(LOGS_DIR / "prospective_forecasts_log.csv", active_hist)
        if not sc_df.empty:
            st.dataframe(sc_df, use_container_width=True, hide_index=True)
        else:
            st.info("Realized accuracy scores appear automatically once future DGHS dates match logged forecasts.")
