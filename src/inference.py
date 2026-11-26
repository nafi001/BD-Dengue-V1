"""
Production Inference Engine for Bangladesh Dengue Multi-Horizon Forecasting & Outbreak Detection.
Uses frozen pooled_lgb_gjr_garch (h01..h14) + coherent Bottom-Up National reconciliation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import holidays
from scipy.stats import t as student_t

QUANTILES = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
Q_COLS = ["q025", "q050", "q100", "q250", "q500", "q750", "q900", "q950", "q975"]
DIVISIONS = ["dhaka", "chattogram", "khulna", "rajshahi", "rangpur", "barisal", "sylhet", "mymensingh"]
COVARIATE_COLS = [
    "Urban_T_mean", "Urban_T_min", "Urban_T_max", "Urban_DTR",
    "Urban_RH", "Urban_Dew_Point", "Urban_Wind_Speed", "Urban_Surface_Pressure",
    "Urban_Rainfall", "Urban_LST_Day", "Urban_LST_Night", "Urban_NDVI", "Urban_NDWI",
]
OPTICAL_COLS = ["Urban_LST_Day", "Urban_LST_Night", "Urban_NDVI", "Urban_NDWI"]
TARGET_LAGS = [0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42]
CLIMATE_LAGS_ALL = [0, 7, 14, 21, 28, 35, 42, 49, 56, 60]
SHORT_LAG_VARS = ["Urban_Rainfall", "Urban_T_mean", "Urban_RH"]
SHORT_LAGS = [1, 2, 3, 4, 5, 6]
ROLL_WINDOWS = [7, 14, 28, 56]
ANOMALY_VARS = ["Urban_T_mean", "Urban_RH", "Urban_Dew_Point", "Urban_Rainfall", "Urban_NDWI"]


def _causal_streak(binary_arr: np.ndarray) -> np.ndarray:
    out = np.zeros(len(binary_arr), dtype=np.int16)
    s = 0
    for i, v in enumerate(binary_arr):
        s = s + 1 if v == 1 else 0
        out[i] = s
    return out


def _causal_seasonal_anomaly(series: pd.Series, dates: pd.Series) -> np.ndarray:
    s = series.astype(float).values
    weeks = dates.dt.isocalendar().week.astype(int).values
    n = len(s)
    anoms = np.zeros(n, dtype=np.float32)
    week_hist = {}
    r_sum, r_cnt = 0.0, 0
    for i in range(n):
        if i >= 300:
            week_hist.setdefault(weeks[i - 300], []).append(s[i - 300])
        w = weeks[i]
        base = float(np.mean(week_hist[w])) if (w in week_hist and week_hist[w]) else (r_sum / r_cnt if r_cnt > 0 else s[i])
        anoms[i] = s[i] - base
        r_sum += s[i]
        r_cnt += 1
    return anoms


def build_calendar_row(target_date: pd.Timestamp) -> dict:
    dow = int(target_date.dayofweek)
    doy = int(target_date.dayofyear)
    month = int(target_date.month)
    bd_hols = holidays.country_holidays("BD", years=[target_date.year - 1, target_date.year, target_date.year + 1])
    is_hol = int(target_date.date() in bd_hols)
    prev_hol = int((target_date - pd.Timedelta(days=1)).date() in bd_hols)

    row = {f"cal_dow_{d}": int(dow == d) for d in range(7)}
    row.update({
        "cal_dow_sin": float(np.sin(2 * np.pi * dow / 7.0)),
        "cal_dow_cos": float(np.cos(2 * np.pi * dow / 7.0)),
        "cal_doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "cal_doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "cal_month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        "cal_month_cos": float(np.cos(2 * np.pi * month / 12.0)),
        "cal_is_friday": int(dow == 4),
        "cal_is_bd_weekend": int(dow in (4, 5)),
        "cal_is_bd_public_holiday": is_hol,
        "cal_is_day_after_holiday": prev_hol,
        "cal_phase_winter": int(month in (12, 1, 2)),
        "cal_phase_pre_monsoon": int(month in (3, 4, 5)),
        "cal_phase_monsoon": int(month in (6, 7, 8, 9)),
        "cal_phase_post_monsoon": int(month in (10, 11)),
    })
    return row


def extract_latest_origin_features(div_df: pd.DataFrame) -> tuple:
    """Strictly causal feature extraction for the latest origin date t in div_df."""
    df = div_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    # Use tail 450 days for fast inference while supporting 60d lags + 300d causal anomaly
    df = df.tail(450).reset_index(drop=True)

    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    df = df.set_index("Date").reindex(full_range).rename_axis("Date").reset_index()

    for col in OPTICAL_COLS:
        if f"{col}_is_missing" not in df.columns:
            is_m = df[col].isna()
            df[f"{col}_is_missing"] = is_m.astype(np.int8)
            ds, cnt, seen = [], 0, False
            for m in is_m.values:
                if not m:
                    cnt, seen = 0, True
                else:
                    cnt = cnt + 1 if seen else -1
                ds.append(cnt)
            df[f"{col}_days_since_valid"] = np.array(ds, dtype=np.int16)

    df["dengue"] = pd.to_numeric(df["dengue"], errors="coerce").ffill().fillna(0.0).clip(lower=0.0)
    for col in COVARIATE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill()
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    lst_flag = (df["Urban_LST_Day"] < 5.0).astype(np.int8)
    if lst_flag.any():
        df.loc[df["Urban_LST_Day"] < 5.0, "Urban_LST_Day"] = np.nan
        df["Urban_LST_Day"] = df["Urban_LST_Day"].ffill().fillna(25.0)

    y = df["dengue"].astype(np.float32)
    ln_y = np.log1p(y).astype(np.float32)
    feats = {}

    for lag in TARGET_LAGS:
        feats[f"dengue_lag{lag}"] = y.shift(lag)
        feats[f"ln_dengue_lag{lag}"] = ln_y.shift(lag)
    for w in ROLL_WINDOWS:
        ry, rln = y.rolling(w, min_periods=1), ln_y.rolling(w, min_periods=1)
        for st, sy, sl in [
            ("mean", ry.mean(), rln.mean()), ("std", ry.std().fillna(0), rln.std().fillna(0)),
            ("sum", ry.sum(), rln.sum()), ("max", ry.max(), rln.max()), ("min", ry.min(), rln.min()),
        ]:
            feats[f"dengue_r{st}_{w}d"] = sy
            feats[f"ln_dengue_r{st}_{w}d"] = sl

    rm7, rm14, rm28 = feats["dengue_rmean_7d"], feats["dengue_rmean_14d"], feats["dengue_rmean_28d"]
    feats["dengue_growth_7d"] = (rm7 + 1.0) / (rm7.shift(7) + 1.0) - 1.0
    feats["dengue_growth_14d"] = (rm14 + 1.0) / (rm14.shift(14) + 1.0) - 1.0
    feats["ln_dengue_diff_7d"] = ln_y - ln_y.shift(7)
    feats["ln_dengue_diff_14d"] = ln_y - ln_y.shift(14)
    feats["dengue_wow_diff"] = y - y.shift(7)
    feats["dengue_accel_7d"] = feats["dengue_wow_diff"] - feats["dengue_wow_diff"].shift(7)
    feats["dengue_surge_ratio_7_28"] = (rm7 + 1.0) / (rm28 + 1.0)
    feats["dengue_cv_14d"] = feats["dengue_rstd_14d"] / (rm14 + 1.0)
    feats["dengue_cv_28d"] = feats["dengue_rstd_28d"] / (rm28 + 1.0)

    for col in COVARIATE_COLS:
        x = df[col].astype(np.float32)
        lags_use = sorted(set(CLIMATE_LAGS_ALL + (SHORT_LAGS if col in SHORT_LAG_VARS else [])))
        for lag in lags_use:
            feats[f"{col}_lag{lag}"] = x.shift(lag)
        for w in ROLL_WINDOWS:
            r = x.rolling(w, min_periods=1)
            feats[f"{col}_rmean_{w}d"] = r.mean()
            feats[f"{col}_rstd_{w}d"] = r.std().fillna(0)
            feats[f"{col}_rmax_{w}d"] = r.max()
            feats[f"{col}_rmin_{w}d"] = r.min()
            if col == "Urban_Rainfall":
                feats[f"{col}_rsum_{w}d"] = r.sum()

    t_m, t_d, rh = df["Urban_T_mean"], df["Urban_Dew_Point"], df["Urban_RH"]
    rain, wind = df["Urban_Rainfall"], df["Urban_Wind_Speed"]
    lst_d, lst_n, ndwi = df["Urban_LST_Day"], df["Urban_LST_Night"], df["Urban_NDWI"]

    vpd = np.maximum(0.6108 * np.exp((17.27 * t_m) / (t_m + 237.3)) - 0.6108 * np.exp((17.27 * t_d) / (t_d + 237.3)), 0.0)
    aedes_s = ((t_m >= 22.0) & (t_m <= 32.0)).astype(np.float32)
    gdd = np.maximum(t_m - 15.0, 0.0)
    heavy, wet, dry = (rain > 50.0).astype(np.int8), (rain >= 1.0).astype(np.int8), (rain < 1.0).astype(np.int8)
    low_w = (wind < 2.5).astype(np.float32)

    feats.update({
        "bio_vpd_kpa": vpd,
        "bio_vpd_rmean_14d": vpd.rolling(14, min_periods=1).mean(),
        "bio_vpd_rmean_28d": vpd.rolling(28, min_periods=1).mean(),
        "bio_aedes_thermal_suit": aedes_s,
        "bio_aedes_suit_days_14d": aedes_s.rolling(14, min_periods=1).sum(),
        "bio_aedes_suit_days_28d": aedes_s.rolling(28, min_periods=1).sum(),
        "bio_aedes_temp_opt_dev": np.abs(t_m - 28.0),
        "bio_gdd_15": gdd,
        "bio_gdd_15_rsum_14d": gdd.rolling(14, min_periods=1).sum(),
        "bio_gdd_15_rsum_28d": gdd.rolling(28, min_periods=1).sum(),
        "bio_heat_humidity_inter": t_m * rh,
        "bio_rain_x_temp": rain * t_m,
        "bio_rain_x_ndwi": rain * ndwi,
        "bio_lst_dtr": lst_d - lst_n,
        "bio_low_wind_flag": low_w,
        "bio_low_wind_days_14d": low_w.rolling(14, min_periods=1).sum(),
        "bio_heavy_rain_gt50mm": heavy.astype(np.float32),
        "bio_heavy_rain_days_14d": heavy.rolling(14, min_periods=1).sum(),
        "bio_heavy_rain_days_28d": heavy.rolling(28, min_periods=1).sum(),
        "bio_wet_days_7d": wet.rolling(7, min_periods=1).sum(),
        "bio_wet_days_14d": wet.rolling(14, min_periods=1).sum(),
        "bio_wet_days_28d": wet.rolling(28, min_periods=1).sum(),
        "bio_consec_wet_days": pd.Series(_causal_streak(wet.values), index=df.index),
        "bio_consec_dry_days": pd.Series(_causal_streak(dry.values), index=df.index),
    })

    for col in ANOMALY_VARS:
        feats[f"{col}_causal_anom"] = pd.Series(_causal_seasonal_anomaly(df[col], df["Date"]), index=df.index)

    for fc in [c for c in df.columns if c.endswith("_is_missing") or c.endswith("_days_since_valid")]:
        if fc != "dengue_is_missing":
            feats[fc] = df[fc]
    feats["Urban_LST_Day_cloud_outlier_flag"] = lst_flag

    last_i = len(df) - 1
    origin_date = pd.Timestamp(df.loc[last_i, "Date"])
    orig_dict = {k: float(v.iloc[last_i]) if pd.notna(v.iloc[last_i]) else 0.0 for k, v in feats.items()}
    y_log_map = dict(zip(df["Date"], ln_y.values.astype(float)))
    return origin_date, orig_dict, y_log_map


def forecast_all_divisions_and_national(history_df: pd.DataFrame, artifacts_dir: Path) -> pd.DataFrame:
    """Generate 1–14d probabilistic forecasts for all 8 divisions + coherent Bottom-Up National."""
    artifacts_dir = Path(artifacts_dir)
    R_train = np.load(artifacts_dir / "cross_div_corr_R_train.npy")

    div_origin_cache = {}
    for div in DIVISIONS:
        sub = history_df[history_df["ADM1_NAME"].str.lower() == div]
        div_origin_cache[div] = extract_latest_origin_features(sub)

    rows = []
    for h in range(1, 15):
        art = joblib.load(artifacts_dir / f"pooled_lgb_gjr_garch_h{h:02d}_combined.joblib")
        est = art["estimator"]
        all_feat_cols = art["all_feat_cols"]
        k_season = 7 * int(np.ceil(h / 7.0))

        h_div_preds = {}
        for div in DIVISIONS:
            orig_dt, orig_feats, y_log_map = div_origin_cache[div]
            targ_dt = orig_dt + pd.Timedelta(days=h)
            cal_row = build_calendar_row(targ_dt)

            row_vals = dict(orig_feats)
            row_vals.update(cal_row)
            for d_oh in DIVISIONS:
                row_vals[f"div_{d_oh}"] = 1.0 if d_oh == div else 0.0
            row_vals["div_train_log_mean"] = float(art["div_stats"]["mean"][div])
            row_vals["div_train_log_std"] = float(art["div_stats"]["std"][div])

            x_vec = np.array([[row_vals.get(c, 0.0) for c in all_feat_cols]], dtype=np.float32)
            anc_log = float(y_log_map.get(targ_dt - pd.Timedelta(days=k_season), 0.0))
            max_cap = float(art["cap_by_div"][div])
            pred_log = float(np.clip(anc_log + float(est.predict(x_vec)[0]), 0.0, max_cap))
            pred_raw = float(np.clip(np.expm1(pred_log), 0.0, None))

            gp = art["garch_by_div"][div]
            persist = gp["alpha"] + 0.5 * gp["gamma"] + gp["beta"]
            var_h = gp["uncond_var"] + (persist ** max(h - 1, 0)) * (gp["init_var"] - gp["uncond_var"])
            sig_h = float(np.sqrt(np.clip(var_h, 1e-4, 4.0)))
            nu = gp["nu"]
            t_q = student_t.ppf(QUANTILES, df=nu) * np.sqrt((nu - 2.0) / nu)
            q_raw = np.sort(np.clip(np.expm1(np.clip(pred_log + sig_h * t_q, 0.0, max_cap + 1.5)), 0.0, None))

            rec = {
                "scope": div, "model": "pooled_lgb_gjr_garch", "horizon": h,
                "origin_date": orig_dt.strftime("%Y-%m-%d"),
                "target_date": targ_dt.strftime("%Y-%m-%d"),
                "y_pred": round(pred_raw, 2),
            }
            for qi, qc in enumerate(Q_COLS):
                rec[qc] = round(float(q_raw[qi]), 2)
            rows.append(rec)
            h_div_preds[div] = rec

        orig_nat = h_div_preds["dhaka"]["origin_date"]
        targ_nat = h_div_preds["dhaka"]["target_date"]
        med_sum = sum(h_div_preds[d]["q500"] for d in DIVISIONS)
        pt_sum = sum(h_div_preds[d]["y_pred"] for d in DIVISIONS)
        nat_q = []
        for qv, qc in zip(QUANTILES, Q_COLS):
            if abs(qv - 0.50) < 1e-6:
                nat_q.append(med_sum)
            else:
                if qv > 0.50:
                    w = np.array([max(h_div_preds[d][qc] - h_div_preds[d]["q500"], 0.0) for d in DIVISIONS])
                    sign = 1.0
                else:
                    w = np.array([max(h_div_preds[d]["q500"] - h_div_preds[d][qc], 0.0) for d in DIVISIONS])
                    sign = -1.0
                hw = float(np.sqrt(max(float(w @ R_train @ w), 0.0)))
                nat_q.append(max(0.0, med_sum + sign * hw))
        nat_q = np.sort(nat_q)

        nat_rec = {
            "scope": "national", "model": "pooled_lgb_gjr_garch_bottom_up", "horizon": h,
            "origin_date": orig_nat, "target_date": targ_nat, "y_pred": round(pt_sum, 2),
        }
        for qi, qc in enumerate(Q_COLS):
            nat_rec[qc] = round(float(nat_q[qi]), 2)
        rows.append(nat_rec)

    return pd.DataFrame(rows)


def compute_weekly_outbreak_status(fc_df: pd.DataFrame, thresh_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-7d & 8-14d forecasts into weekly totals & classify alert tier."""
    rows = []
    for scope in DIVISIONS + ["national"]:
        sub = fc_df[fc_df["scope"] == scope].sort_values("horizon")
        for label, h_range in [("Week +1 (Days 1-7)", range(1, 8)), ("Week +2 (Days 8-14)", range(8, 15))]:
            w_sub = sub[sub["horizon"].isin(h_range)]
            wk_end = pd.to_datetime(w_sub["target_date"]).max()
            epi_wk = min(int(wk_end.isocalendar().week), 52)
            t_row = thresh_df[(thresh_df["scope"] == scope) & (thresh_df["epi_week"] == epi_wk)].iloc[0]

            pred_sum = float(w_sub["y_pred"].sum())
            q90_sum = float(w_sub["q900"].sum())
            p75, p90, m2sd = float(t_row["thresh_p75"]), float(t_row["thresh_p90"]), float(t_row["thresh_mean_2sd"])

            if pred_sum > m2sd:
                status = "EPIDEMIC (> Mean+2SD)"
            elif pred_sum > p90:
                status = "HIGH ALERT (> P90)"
            elif pred_sum > p75:
                status = "ALERT (> P75)"
            elif q90_sum > p90:
                status = "WATCH (Upper 90% > P90)"
            else:
                status = "NORMAL"

            rows.append({
                "scope": scope, "lead_window": label, "week_ending": wk_end.strftime("%Y-%m-%d"),
                "epi_week": epi_wk, "projected_weekly_cases": round(pred_sum, 1),
                "projected_upper_90": round(q90_sum, 1), "thresh_p75": round(p75, 1),
                "thresh_p90": round(p90, 1), "thresh_mean_2sd": round(m2sd, 1), "alert_status": status,
            })
    return pd.DataFrame(rows)
