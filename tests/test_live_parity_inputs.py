import sys
from collections import deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml import features as feat_eng


def test_build_live_funding_frame_uses_real_record_timestamps():
    df5_live = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-21 00:00:00", periods=10, freq="5min", tz="UTC"),
            "open": [1.0] * 10,
            "high": [1.0] * 10,
            "low": [1.0] * 10,
            "close": [1.0] * 10,
            "volume": [1.0] * 10,
        }
    )
    funding_records = deque(
        [
            {"timestamp": pd.Timestamp("2026-04-20 16:00:00", tz="UTC"), "funding_rate": 0.001},
            {"timestamp": pd.Timestamp("2026-04-21 00:00:00", tz="UTC"), "funding_rate": 0.002},
        ],
        maxlen=24,
    )

    funding_df = feat_eng._build_live_funding_frame(df5_live, 0.002, funding_records)

    assert funding_df["timestamp"].tolist() == [
        pd.Timestamp("2026-04-20 16:00:00", tz="UTC"),
        pd.Timestamp("2026-04-21 00:00:00", tz="UTC"),
    ]
    assert funding_df["funding_rate"].tolist() == [0.001, 0.002]
