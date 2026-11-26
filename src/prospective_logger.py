"""
Prospective Forecast Logger, Realized Accuracy Scorer & Manual QA Release Sign-off Log.
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

BD_TZ = timezone(timedelta(hours=6))


def append_prospective_forecasts(fc_df: pd.DataFrame, log_path: Path, release_id: str) -> int:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = fc_df.copy()
    out["release_id"] = release_id
    out["logged_at_bd"] = datetime.now(BD_TZ).strftime("%Y-%m-%d %H:%M:%S")

    if log_path.exists() and log_path.stat().st_size > 0:
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, out], ignore_index=True)
        combined = combined.drop_duplicates(subset=["scope", "origin_date", "target_date", "horizon", "model"], keep="last")
        combined.to_csv(log_path, index=False)
    else:
        out.to_csv(log_path, index=False)
    return len(out)


def append_qa_signoff(qa_record: dict, qa_log_path: Path) -> None:
    qa_log_path = Path(qa_log_path)
    qa_log_path.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(qa_record)
    rec["signed_at_bd"] = datetime.now(BD_TZ).strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame([rec])
    header = not (qa_log_path.exists() and qa_log_path.stat().st_size > 0)
    df_new.to_csv(qa_log_path, mode="a", header=header, index=False)


def score_logged_forecasts(log_path: Path, actuals_df: pd.DataFrame) -> pd.DataFrame:
    log_path = Path(log_path)
    if not (log_path.exists() and log_path.stat().st_size > 0):
        return pd.DataFrame()

    fc = pd.read_csv(log_path)
    act = actuals_df.copy()
    act["Date"] = pd.to_datetime(act["Date"]).dt.strftime("%Y-%m-%d")
    act["scope"] = act["ADM1_NAME"].str.lower()
    nat_act = act.groupby("Date", as_index=False)["dengue"].sum()
    nat_act["scope"] = "national"
    act_all = pd.concat([act[["scope", "Date", "dengue"]], nat_act[["scope", "Date", "dengue"]]], ignore_index=True)

    merged = fc.merge(act_all, left_on=["scope", "target_date"], right_on=["scope", "Date"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["abs_err"] = np.abs(merged["dengue"] - merged["y_pred"])
    merged["in_80"] = ((merged["dengue"] >= merged["q100"]) & (merged["dengue"] <= merged["q900"])).astype(float) * 100.0
    merged["in_95"] = ((merged["dengue"] >= merged["q025"]) & (merged["dengue"] <= merged["q975"])).astype(float) * 100.0

    return (
        merged.groupby("scope", as_index=False)
        .agg(n_scored=("dengue", "count"), MAE=("abs_err", "mean"),
             Cov_80_pct=("in_80", "mean"), Cov_95_pct=("in_95", "mean"))
        .round(2)
    )
