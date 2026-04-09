import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/nebula/nyxtest4')

from ml.features import compute_atr14, FEATURE_COLS
import config as cfg


def make_ohlcv(n=100, seed=42):
    rng = np.random.default_rng(seed)
    close = 50000 + np.cumsum(rng.normal(0, 100, n))
    open_ = close + rng.normal(0, 50, n)
    high = np.maximum(close, open_) + rng.uniform(0, 100, n)
    low = np.minimum(close, open_) - rng.uniform(0, 100, n)
    vol = rng.uniform(10, 100, n)
    ts = pd.date_range('2025-01-01', periods=n, freq='5min', tz='UTC')
    return pd.DataFrame({'timestamp': ts, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': vol})


def test_feature_count():
    assert len(FEATURE_COLS) == 22


def test_feature_order():
    expected = ['body_ratio_n1', 'body_ratio_n2', 'body_ratio_n3',
                'upper_wick_n1', 'upper_wick_n2', 'lower_wick_n1', 'lower_wick_n2',
                'volume_ratio_n1', 'volume_ratio_n2',
                'body_ratio_15m', 'dir_15m', 'volume_ratio_15m',
                'body_ratio_1h', 'dir_1h', 'ema9_slope_1h',
                'funding_rate', 'funding_zscore',
                'delta_ratio', 'cvd_delta', 'cvd_5', 'cvd_20', 'cvd_trend']
    assert FEATURE_COLS == expected


def test_atr14_formula():
    df = make_ohlcv(50)
    atr = compute_atr14(df)
    assert (atr.dropna() > 0).all()
    assert atr.iloc[:13].isna().all()
    assert not pd.isna(atr.iloc[14])


def test_no_lookahead_body_ratio_n1():
    df = make_ohlcv(50)
    atr = compute_atr14(df)
    expected = (df['close'].iloc[19] - df['open'].iloc[19]) / atr.iloc[19]
    br_series = (df['close'].shift(1) - df['open'].shift(1)) / atr.shift(1)
    assert abs(br_series.iloc[20] - expected) < 1e-10


def test_cvd_proxy_formula():
    high, low, close, vol = 100.0, 90.0, 95.0, 1000.0
    buy = vol * (close - low) / (high - low)
    sell = vol * (high - close) / (high - low)
    assert abs(buy - 500.0) < 1e-6
    assert abs(sell - 500.0) < 1e-6


def test_merge_asof_no_future_leak():
    ts_5m = pd.Timestamp('2025-01-01 09:00:00', tz='UTC')
    ts_15m_future = pd.Timestamp('2025-01-01 09:15:00', tz='UTC')
    ts_15m_current = pd.Timestamp('2025-01-01 09:00:00', tz='UTC')
    ts_15m_past = pd.Timestamp('2025-01-01 08:45:00', tz='UTC')
    left = pd.DataFrame({'ts_n1': [ts_5m]})
    right = pd.DataFrame({'timestamp': [ts_15m_past, ts_15m_current, ts_15m_future], 'val': [1, 2, 3]})
    merged = pd.merge_asof(left, right, left_on='ts_n1', right_on='timestamp', direction='backward')
    assert merged['val'].iloc[0] == 2


def test_train_val_test_split():
    n = 1000
    train_end = int(n * 0.75)
    val_start = int(train_end * 0.80)
    assert train_end == 750
    assert val_start == 600
    assert n - train_end == 250
    assert train_end - val_start == 150


def test_default_threshold_matches_blueprint():
    """Blueprint Section 9: recommended threshold is 0.535.
    This test will catch any future accidental regression of the default."""
    assert cfg.ML_DEFAULT_THRESHOLD == 0.535, (
        f"ML_DEFAULT_THRESHOLD is {cfg.ML_DEFAULT_THRESHOLD}, expected 0.535 "
        "(Blueprint Section 9 recommended threshold)"
    )


def test_asof_backward_vectorized_matches_searchsorted():
    """_asof_backward (now pd.merge_asof) must produce identical results to
    the previous searchsorted row-loop implementation for all call sites:
    15m merge, 1h merge, funding merge, and CVD merge."""
    import sys
    sys.path.insert(0, '/home/nebula/nyxtest4')
    from ml.features import _asof_backward

    rng = np.random.default_rng(0)
    n_left = 200
    n_right = 50

    # Build a right-side DataFrame with sorted timestamps and two value columns
    right_ts = pd.date_range('2025-01-01', periods=n_right, freq='15min', tz='UTC')
    right = pd.DataFrame({
        'timestamp': right_ts,
        'val_a': rng.uniform(0, 1, n_right),
        'val_b': rng.uniform(100, 200, n_right),
    })

    # Build left timestamps — denser than right, some before first right row (should give NaN)
    left_ts = pd.date_range('2024-12-31 23:00', periods=n_left, freq='5min', tz='UTC')
    left_series = pd.Series(left_ts)

    # Run the vectorized implementation
    result = _asof_backward(left_series, right, ['val_a', 'val_b'])

    # Re-implement the original searchsorted logic inline for reference.
    # Use microseconds (us) throughout — pandas 2.x stores datetime64[us],
    # so .values.view(int64) gives us-since-epoch. Convert left ts the same way.
    right_ts_us = right['timestamp'].values.view(np.int64)  # datetime64[us] -> int64 us
    expected_a = np.full(n_left, np.nan)
    expected_b = np.full(n_left, np.nan)
    for i, ts in enumerate(left_series):
        if pd.isna(ts):
            continue
        # Convert to microseconds: Timestamp.value is ns, divide by 1000
        ts_us = pd.Timestamp(ts).value // 1000
        idx = np.searchsorted(right_ts_us, ts_us, side='right') - 1
        if idx >= 0 and right_ts_us[idx] <= ts_us:
            expected_a[i] = right['val_a'].iloc[idx]
            expected_b[i] = right['val_b'].iloc[idx]

    # Results must be bit-for-bit identical (same float values, not just close)
    got_a = result['val_a'].values
    got_b = result['val_b'].values
    nan_mask_a = np.isnan(expected_a)
    nan_mask_b = np.isnan(expected_b)
    assert np.array_equal(nan_mask_a, np.isnan(got_a)), "NaN positions differ for val_a"
    assert np.array_equal(nan_mask_b, np.isnan(got_b)), "NaN positions differ for val_b"
    np.testing.assert_array_equal(got_a[~nan_mask_a], expected_a[~nan_mask_a])
    np.testing.assert_array_equal(got_b[~nan_mask_b], expected_b[~nan_mask_b])


def test_volume_ratio_n1_excludes_self_from_mean():
    """volume_ratio_n1 = volume[i-1] / mean(volume[i-2]..volume[i-21]).
    The N-1 candle must NOT appear in its own rolling mean denominator.
    Training formula: shift(2).rolling(20) at row i = mean of [i-2..i-21].
    Live formula:     vol_series[-22:-2]            = mean of [i-2..i-21].
    Both must be identical — this test verifies the training-side formula.
    """
    df = make_ohlcv(60)
    # Compute training formula
    vol_mean_train = df['volume'].shift(2).rolling(20).mean()
    ratio_train = df['volume'].shift(1) / vol_mean_train

    # Compute live formula manually at the last row
    vol = df['volume'].values
    # Last row index = 59 (i=59), N-1 = index 58, mean window = [57..38]
    live_mean = np.mean(vol[38:58])   # indices 38..57 inclusive = vol[-22:-2] of 60-row array
    live_ratio = vol[58] / live_mean

    train_ratio_last = ratio_train.iloc[59]
    assert abs(train_ratio_last - live_ratio) < 1e-10, (
        f"Train/live volume_ratio_n1 mismatch: train={train_ratio_last:.8f} live={live_ratio:.8f}"
    )
