"""
Automated Data Quality & Surveillance Integrity Audit for Bangladesh Dengue EWS.
"""
import numpy as np
import pandas as pd

REQUIRED_RAW_COLS = [
    "ADM1_NAME", "Date", "dengue",
    "Urban_T_mean", "Urban_T_min", "Urban_T_max", "Urban_DTR",
    "Urban_RH", "Urban_Dew_Point", "Urban_Wind_Speed", "Urban_Surface_Pressure",
    "Urban_Rainfall", "Urban_LST_Day", "Urban_LST_Night", "Urban_NDVI", "Urban_NDWI",
]


def audit_division_dataframe(df: pd.DataFrame, division_name: str, reference_date=None) -> dict:
    """Run pre-inference data quality checks on a single division's daily DataFrame."""
    flags = []
    severity = "PASS"

    missing_cols = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
    if missing_cols:
        return {
            "division": division_name, "status": "FAIL", "n_days": len(df),
            "latest_date": "N/A", "staleness_days": -1, "dengue_missing_14d": -1,
            "lst_cloud_outliers_14d": -1, "is_latest_friday": False,
            "flags": f"Missing columns: {missing_cols}",
        }

    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    bad_dates = int(work["Date"].isna().sum())
    work = work.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")

    latest_dt = work["Date"].max()
    recent_14 = work.tail(14)
    recent_30 = work.tail(30)

    expected_30 = (recent_30["Date"].max() - recent_30["Date"].min()).days + 1
    gap_days = max(0, expected_30 - len(recent_30))
    if gap_days > 0:
        flags.append(f"{gap_days} date gap(s) in last 30d")
        severity = "WARN"

    neg_cnt = int((recent_14["dengue"] < 0).sum())
    miss_dengue = int(recent_14["dengue"].isna().sum())
    if neg_cnt > 0:
        flags.append(f"{neg_cnt} negative admission count(s)")
        severity = "FAIL"
    if miss_dengue > 0:
        flags.append(f"{miss_dengue} missing DGHS day(s) in last 14d")
        severity = "WARN" if miss_dengue <= 2 else "FAIL"

    lst_outliers = int((recent_14["Urban_LST_Day"] < 5.0).sum())
    if lst_outliers > 0:
        flags.append(f"{lst_outliers} cloud-contaminated LST_Day obs (<5C, auto-guarded)")
        if severity == "PASS":
            severity = "WARN"

    opt_miss = int(recent_14[["Urban_NDVI", "Urban_NDWI"]].isna().sum().sum())
    if opt_miss > 4:
        flags.append(f"High optical NDVI/NDWI missingness ({opt_miss} cells in 14d)")
        if severity == "PASS":
            severity = "WARN"

    is_fri = bool(latest_dt.dayofweek == 4)
    if is_fri:
        flags.append("Origin is Friday (typical ~48% DGHS reporting dip; modeled via calendar)")

    staleness = 0
    if reference_date is not None:
        staleness = int((pd.Timestamp(reference_date) - latest_dt).days)
        if staleness > 3:
            flags.append(f"Stale feed ({staleness} days behind reference)")
            if severity == "PASS":
                severity = "WARN"

    if bad_dates > 0:
        flags.append(f"{bad_dates} unparseable date rows dropped")
        if severity == "PASS":
            severity = "WARN"

    return {
        "division": division_name,
        "status": severity,
        "n_days": int(len(work)),
        "latest_date": latest_dt.strftime("%Y-%m-%d"),
        "staleness_days": staleness,
        "dengue_missing_14d": miss_dengue,
        "lst_cloud_outliers_14d": lst_outliers,
        "is_latest_friday": is_fri,
        "flags": "; ".join(flags) if flags else "All checks passed",
    }


def audit_all_divisions(history_df: pd.DataFrame, divisions: list, reference_date=None) -> pd.DataFrame:
    rows = []
    for div in divisions:
        sub = history_df[history_df["ADM1_NAME"].str.lower() == div]
        rows.append(audit_division_dataframe(sub, div, reference_date=reference_date))
    return pd.DataFrame(rows)
