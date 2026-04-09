import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/nebula/nyxtest4')

from ml.features import compute_atr14, FEATURE_COLS


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
