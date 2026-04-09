"""Feature engineering for LightGBM ML strategy — BLUEPRINT sections 4, 5, 6.

ZERO lookahead bias: all features use shift(k>=1). Target uses shift(-1) (future,
only for training labels — never used as a feature).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column order — MUST match exactly (22 features)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "body_ratio_n1", "body_ratio_n2", "body_ratio_n3",
    "upper_wick_n1", "upper_wick_n2",
    "lower_wick_n1", "lower_wick_n2",
    "volume_ratio_n1", "volume_ratio_n2",
    "body_ratio_15m", "dir_15m", "volume_ratio_15m",
    "body_ratio_1h", "dir_1h", "ema9_slope_1h",
    "funding_rate", "funding_zscore",
    "delta_ratio", "cvd_delta", "cvd_5", "cvd_20", "cvd_trend",
]


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    """ATR14 using EWM (BLUEPRINT spec)."""
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()


def _asof_backward(left_ts: pd.Series, right: pd.DataFrame, right_cols: list[str]) -> pd.DataFrame:
    """
    Backward-fill lookup using searchsorted (avoids merge_asof dtype restrictions).
    left_ts: Series of tz-aware timestamps (may contain NaT).
    right: sorted DataFrame with 'timestamp' column + right_cols.
    Returns DataFrame indexed 0..len(left_ts)-1 with right_cols, NaN where no match or NaT.
    """
    n = len(left_ts)
    result = pd.DataFrame(np.nan, index=np.arange(n), columns=right_cols)

    # Convert both to int64 (nanoseconds) for comparison — avoids dtype issues entirely
    right_ts_ns = right["timestamp"].astype(np.int64).values  # sorted ascending

    for i, ts in enumerate(left_ts):
        if pd.isna(ts):
            continue
        ts_ns = pd.Timestamp(ts).value  # int64 nanoseconds
        # searchsorted: find insertion point (right side = first element > ts_ns)
        idx = np.searchsorted(right_ts_ns, ts_ns, side="right") - 1
        if idx >= 0:
            result.iloc[i] = right[right_cols].iloc[idx].values
    return result


def build_features(
    df5: pd.DataFrame,
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    funding: pd.DataFrame,
    cvd: pd.DataFrame,
) -> pd.DataFrame:
    """Build 22 features per BLUEPRINT sections 4-6. Returns df with FEATURE_COLS + 'target'."""

    # Work on copies with clean RangeIndex
    df5 = df5.copy().reset_index(drop=True)
    df15 = df15.copy().reset_index(drop=True)
    df1h = df1h.copy().reset_index(drop=True)
    funding = funding.copy().reset_index(drop=True)
    cvd = cvd.copy().reset_index(drop=True)

    # Sort ascending (should already be sorted, but be safe)
    df5 = df5.sort_values("timestamp").reset_index(drop=True)
    df15 = df15.sort_values("timestamp").reset_index(drop=True)
    df1h = df1h.sort_values("timestamp").reset_index(drop=True)
    funding = funding.sort_values("timestamp").reset_index(drop=True)
    cvd = cvd.sort_values("timestamp").reset_index(drop=True)

    # Normalize all timestamps to ms UTC for consistent merging
    for df in [df5, df15, df1h, funding, cvd]:
        df["timestamp"] = df["timestamp"].astype("datetime64[ms, UTC]")

    # -----------------------------------------------------------------------
    # 5m features — all use shift(k>=1), NEVER shift(0)
    # -----------------------------------------------------------------------
    atr5 = compute_atr14(df5)

    df5["body_ratio_n1"] = (df5["close"].shift(1) - df5["open"].shift(1)) / atr5.shift(1)
    df5["body_ratio_n2"] = (df5["close"].shift(2) - df5["open"].shift(2)) / atr5.shift(2)
    df5["body_ratio_n3"] = (df5["close"].shift(3) - df5["open"].shift(3)) / atr5.shift(3)

    df5["upper_wick_n1"] = (
        df5["high"].shift(1) - df5[["open", "close"]].shift(1).max(axis=1)
    ) / atr5.shift(1)
    df5["upper_wick_n2"] = (
        df5["high"].shift(2) - df5[["open", "close"]].shift(2).max(axis=1)
    ) / atr5.shift(2)

    df5["lower_wick_n1"] = (
        df5[["open", "close"]].shift(1).min(axis=1) - df5["low"].shift(1)
    ) / atr5.shift(1)
    df5["lower_wick_n2"] = (
        df5[["open", "close"]].shift(2).min(axis=1) - df5["low"].shift(2)
    ) / atr5.shift(2)

    vol_mean_n1 = df5["volume"].shift(1).rolling(20).mean()
    df5["volume_ratio_n1"] = df5["volume"].shift(1) / vol_mean_n1
    vol_mean_n2 = df5["volume"].shift(2).rolling(20).mean()
    df5["volume_ratio_n2"] = df5["volume"].shift(2) / vol_mean_n2

    # ts_n1 = N-1 timestamp (shift by 1 for all multi-tf merges)
    ts_n1 = df5["timestamp"].shift(1)

    # -----------------------------------------------------------------------
    # 15m features — merge_asof backward on ts_n1
    # -----------------------------------------------------------------------
    atr15 = compute_atr14(df15)
    df15["body_ratio_15m"] = (df15["close"] - df15["open"]) / atr15
    df15["dir_15m"] = np.sign(df15["close"] - df15["open"])
    df15["volume_ratio_15m"] = df15["volume"] / df15["volume"].rolling(20).mean()

    r15 = _asof_backward(ts_n1, df15, ["body_ratio_15m", "dir_15m", "volume_ratio_15m"])
    df5["body_ratio_15m"] = r15["body_ratio_15m"].values
    df5["dir_15m"] = r15["dir_15m"].values
    df5["volume_ratio_15m"] = r15["volume_ratio_15m"].values

    # -----------------------------------------------------------------------
    # 1h features — same merge_asof pattern
    # -----------------------------------------------------------------------
    atr1h = compute_atr14(df1h)
    df1h["body_ratio_1h"] = (df1h["close"] - df1h["open"]) / atr1h
    df1h["dir_1h"] = np.sign(df1h["close"] - df1h["open"])
    ema9 = df1h["close"].ewm(span=9, adjust=False).mean()
    df1h["ema9_slope_1h"] = (ema9 - ema9.shift(1)) / atr1h

    r1h = _asof_backward(ts_n1, df1h, ["body_ratio_1h", "dir_1h", "ema9_slope_1h"])
    df5["body_ratio_1h"] = r1h["body_ratio_1h"].values
    df5["dir_1h"] = r1h["dir_1h"].values
    df5["ema9_slope_1h"] = r1h["ema9_slope_1h"].values

    # -----------------------------------------------------------------------
    # Funding features
    # -----------------------------------------------------------------------
    funding["funding_zscore"] = (
        funding["funding_rate"] - funding["funding_rate"].rolling(24).mean()
    ) / funding["funding_rate"].rolling(24).std()
    funding.loc[funding["funding_rate"].rolling(24).std() == 0, "funding_zscore"] = np.nan

    rf = _asof_backward(ts_n1, funding, ["funding_rate", "funding_zscore"])
    df5["funding_rate"] = rf["funding_rate"].values
    df5["funding_zscore"] = rf["funding_zscore"].values

    # -----------------------------------------------------------------------
    # CVD features — merge on exact timestamp then shift(1)
    # -----------------------------------------------------------------------
    # Normalize CVD timestamps to ms to match df5
    cvd["timestamp"] = cvd["timestamp"].astype("datetime64[ms, UTC]")

    atr_cvd = compute_atr14(cvd)
    cvd["delta"] = cvd["buy_vol"] - cvd["sell_vol"]
    denom = (cvd["buy_vol"] + cvd["sell_vol"]).clip(lower=1e-8)
    cvd["delta_ratio_raw"] = cvd["buy_vol"] / denom
    cvd["cvd_delta_raw"] = cvd["delta"] / atr_cvd
    cvd["cvd_5_raw"] = cvd["delta"].rolling(5).sum() / atr_cvd
    cvd["cvd_20_raw"] = cvd["delta"].rolling(20).sum() / atr_cvd
    cvd["cvd_trend_raw"] = cvd["cvd_5_raw"] - cvd["cvd_20_raw"]

    # Merge CVD on df5['timestamp'] directly (same 5m grid), then shift result by 1
    cvd_cols = ["delta_ratio_raw", "cvd_delta_raw", "cvd_5_raw", "cvd_20_raw", "cvd_trend_raw"]
    rcvd = _asof_backward(df5["timestamp"], cvd, cvd_cols)

    # shift by 1 to prevent lookahead (use N-1 CVD value)
    df5["delta_ratio"] = rcvd["delta_ratio_raw"].shift(1).values
    df5["cvd_delta"] = rcvd["cvd_delta_raw"].shift(1).values
    df5["cvd_5"] = rcvd["cvd_5_raw"].shift(1).values
    df5["cvd_20"] = rcvd["cvd_20_raw"].shift(1).values
    df5["cvd_trend"] = rcvd["cvd_trend_raw"].shift(1).values

    # -----------------------------------------------------------------------
    # Target: 1 if close[i+1] > close[i] (future label, NOT a feature)
    # -----------------------------------------------------------------------
    df5["target"] = (df5["close"].shift(-1) > df5["close"]).astype(int)

    # -----------------------------------------------------------------------
    # Drop rows with any NaN in features or target, return feature cols + target
    # -----------------------------------------------------------------------
    all_cols = FEATURE_COLS + ["target"]
    df_out = df5[all_cols].dropna()
    log.info("build_features: %d rows after dropna (started with %d)", len(df_out), len(df5))
    return df_out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Live feature computation
# ---------------------------------------------------------------------------

def build_live_features(
    df5_live: pd.DataFrame,
    df15_live: pd.DataFrame,
    df1h_live: pd.DataFrame,
    funding_rate_float: float | None,
    funding_buffer: deque,
    cvd_live: pd.DataFrame,
) -> "np.ndarray | None":
    """
    Build a single feature row (shape 1×22) for live inference.
    Returns None if ATR warmup not satisfied (fewer than 14 candles).
    """
    # Validate ATR warmup
    if len(df5_live) < 14:
        return None

    df5 = df5_live.copy().reset_index(drop=True)
    df15 = df15_live.copy().reset_index(drop=True)
    df1h = df1h_live.copy().reset_index(drop=True)
    cvd = cvd_live.copy().reset_index(drop=True) if cvd_live is not None and len(cvd_live) > 0 else pd.DataFrame()

    # Normalize timestamps
    for df in [df5, df15, df1h]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ms, UTC]")

    atr5 = compute_atr14(df5)
    if atr5.iloc[-1] is None or pd.isna(atr5.iloc[-1]):
        return None

    # 5m features using last row (index -1 = current candle N)
    # We use shift(1) = N-1
    def safe(series, k=1):
        idx = len(series) - 1 - k
        return series.iloc[idx] if idx >= 0 else np.nan

    atr5_val = safe(atr5, 1)

    body_ratio_n1 = (safe(df5["close"], 1) - safe(df5["open"], 1)) / atr5_val
    body_ratio_n2 = (safe(df5["close"], 2) - safe(df5["open"], 2)) / safe(atr5, 2)
    body_ratio_n3 = (safe(df5["close"], 3) - safe(df5["open"], 3)) / safe(atr5, 3)

    upper_wick_n1 = (safe(df5["high"], 1) - max(safe(df5["open"], 1), safe(df5["close"], 1))) / atr5_val
    upper_wick_n2 = (safe(df5["high"], 2) - max(safe(df5["open"], 2), safe(df5["close"], 2))) / safe(atr5, 2)
    lower_wick_n1 = (min(safe(df5["open"], 1), safe(df5["close"], 1)) - safe(df5["low"], 1)) / atr5_val
    lower_wick_n2 = (min(safe(df5["open"], 2), safe(df5["close"], 2)) - safe(df5["low"], 2)) / safe(atr5, 2)

    vol_window = min(20, len(df5))
    vol_series = df5["volume"].values
    vol_ratio_n1 = vol_series[-2] / np.mean(vol_series[-vol_window - 1:-1]) if vol_window > 0 and len(vol_series) >= 2 else np.nan
    vol_ratio_n2 = vol_series[-3] / np.mean(vol_series[-vol_window - 2:-2]) if vol_window > 0 and len(vol_series) >= 3 else np.nan

    # 15m features
    if len(df15) >= 14:
        atr15 = compute_atr14(df15)
        ts_n1 = df5["timestamp"].iloc[-2] if len(df5) >= 2 else None
        if ts_n1 is not None and not pd.isna(ts_n1):
            # Find the last 15m candle at or before ts_n1
            mask15 = df15["timestamp"] <= ts_n1
            if mask15.any():
                idx15 = df15[mask15].index[-1]
                atr15_val = atr15.iloc[idx15]
                if pd.notna(atr15_val) and atr15_val > 0:
                    body_ratio_15m = (df15["close"].iloc[idx15] - df15["open"].iloc[idx15]) / atr15_val
                    dir_15m = np.sign(df15["close"].iloc[idx15] - df15["open"].iloc[idx15])
                    vol15_window = min(20, idx15 + 1)
                    vol_ratio_15m = df15["volume"].iloc[idx15] / df15["volume"].iloc[max(0, idx15 - vol15_window):idx15].mean() if vol15_window > 0 else np.nan
                else:
                    body_ratio_15m = dir_15m = vol_ratio_15m = np.nan
            else:
                body_ratio_15m = dir_15m = vol_ratio_15m = np.nan
        else:
            body_ratio_15m = dir_15m = vol_ratio_15m = np.nan
    else:
        body_ratio_15m = dir_15m = vol_ratio_15m = np.nan

    # 1h features
    if len(df1h) >= 14:
        atr1h = compute_atr14(df1h)
        ts_n1 = df5["timestamp"].iloc[-2] if len(df5) >= 2 else None
        if ts_n1 is not None and not pd.isna(ts_n1):
            mask1h = df1h["timestamp"] <= ts_n1
            if mask1h.any():
                idx1h = df1h[mask1h].index[-1]
                atr1h_val = atr1h.iloc[idx1h]
                if pd.notna(atr1h_val) and atr1h_val > 0:
                    body_ratio_1h = (df1h["close"].iloc[idx1h] - df1h["open"].iloc[idx1h]) / atr1h_val
                    dir_1h = np.sign(df1h["close"].iloc[idx1h] - df1h["open"].iloc[idx1h])
                    ema9 = df1h["close"].ewm(span=9, adjust=False).mean()
                    ema9_slope_1h = (ema9.iloc[idx1h] - ema9.iloc[idx1h - 1]) / atr1h_val if idx1h > 0 else np.nan
                else:
                    body_ratio_1h = dir_1h = ema9_slope_1h = np.nan
            else:
                body_ratio_1h = dir_1h = ema9_slope_1h = np.nan
        else:
            body_ratio_1h = dir_1h = ema9_slope_1h = np.nan
    else:
        body_ratio_1h = dir_1h = ema9_slope_1h = np.nan

    # Funding features
    if funding_rate_float is not None and len(funding_buffer) > 0:
        buf = list(funding_buffer)
        fr = funding_rate_float
        if len(buf) >= 2:
            mean24 = np.mean(buf)
            std24 = np.std(buf)
            funding_zscore = (fr - mean24) / std24 if std24 > 0 else np.nan
        else:
            funding_zscore = np.nan
    else:
        fr = np.nan
        funding_zscore = np.nan

    # CVD features
    if len(cvd) >= 14 and "buy_vol" in cvd.columns:
        cvd["timestamp"] = pd.to_datetime(cvd["timestamp"], utc=True).astype("datetime64[ms, UTC]")
        atr_cvd = compute_atr14(cvd)
        cvd["delta"] = cvd["buy_vol"] - cvd["sell_vol"]
        denom = (cvd["buy_vol"] + cvd["sell_vol"]).clip(lower=1e-8)
        cvd["delta_ratio_raw"] = cvd["buy_vol"] / denom
        cvd["cvd_delta_raw"] = cvd["delta"] / atr_cvd
        cvd["cvd_5_raw"] = cvd["delta"].rolling(5).sum() / atr_cvd
        cvd["cvd_20_raw"] = cvd["delta"].rolling(20).sum() / atr_cvd
        cvd["cvd_trend_raw"] = cvd["cvd_5_raw"] - cvd["cvd_20_raw"]

        # Use N-2 (shift 1 from last row)
        idx_cvd = len(cvd) - 2
        if idx_cvd >= 0:
            delta_ratio = cvd["delta_ratio_raw"].iloc[idx_cvd]
            cvd_delta = cvd["cvd_delta_raw"].iloc[idx_cvd]
            cvd_5 = cvd["cvd_5_raw"].iloc[idx_cvd]
            cvd_20 = cvd["cvd_20_raw"].iloc[idx_cvd]
            cvd_trend = cvd["cvd_trend_raw"].iloc[idx_cvd]
        else:
            delta_ratio = cvd_delta = cvd_5 = cvd_20 = cvd_trend = np.nan
    else:
        delta_ratio = cvd_delta = cvd_5 = cvd_20 = cvd_trend = np.nan

    row = np.array([[
        body_ratio_n1, body_ratio_n2, body_ratio_n3,
        upper_wick_n1, upper_wick_n2,
        lower_wick_n1, lower_wick_n2,
        vol_ratio_n1, vol_ratio_n2,
        body_ratio_15m, dir_15m, vol_ratio_15m,
        body_ratio_1h, dir_1h, ema9_slope_1h,
        fr, funding_zscore,
        delta_ratio, cvd_delta, cvd_5, cvd_20, cvd_trend,
    ]], dtype=np.float64)

    if np.isnan(row).all():
        return None

    return row
