import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml import data_fetcher


EXPECTED_COLS = ["timestamp", "long_taker_size", "short_taker_size", "open_interest"]


def test_fetch_live_gate_cvd_delegates_to_canonical_fetch_with_frame_anchor(monkeypatch):
    captured = {}

    timestamps = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
    source_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "long_taker_size": range(12),
            "short_taker_size": range(100, 112),
            "open_interest": range(200, 212),
        }
    )

    def fake_fetch_gate_cvd(start_ms: int, end_ms: int) -> pd.DataFrame:
        captured["start_ms"] = start_ms
        captured["end_ms"] = end_ms
        return source_df.copy()

    monkeypatch.setattr(data_fetcher, "fetch_gate_cvd", fake_fetch_gate_cvd)

    anchor_ts = pd.date_range("2026-04-21 12:00:00", periods=20, freq="5min", tz="UTC")
    result = data_fetcher.fetch_live_gate_cvd(limit=5, anchor_timestamps=anchor_ts)

    assert list(result.columns) == EXPECTED_COLS
    assert len(result) == 5
    pd.testing.assert_frame_equal(result.reset_index(drop=True), source_df.tail(5).reset_index(drop=True))

    expected_end_ms = int(anchor_ts.max().timestamp() * 1000) + (5 * 60 * 1000)
    assert captured["end_ms"] == expected_end_ms
    expected_window_candles = 5 + 50
    expected_start_ms = expected_end_ms - (expected_window_candles * 5 * 60 * 1000)
    assert captured["start_ms"] == expected_start_ms


def test_fetch_live_gate_cvd_preserves_empty_schema(monkeypatch):
    empty = pd.DataFrame(columns=EXPECTED_COLS)
    monkeypatch.setattr(data_fetcher, "fetch_gate_cvd", lambda start_ms, end_ms: empty)

    result = data_fetcher.fetch_live_gate_cvd(limit=7)

    assert result.empty
    assert list(result.columns) == EXPECTED_COLS
