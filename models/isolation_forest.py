"""Per-symbol Isolation Forest with an honest first-snapshot warm-up fallback."""

from __future__ import annotations

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from data_collector.yahoo_provider import normalize_symbol
from database.db import query_frame, replace_symbol_frame, upsert_model_registry
from models.common import artifact_path

FEATURES = [
    "volume_delta",
    "volume_zscore",
    "option_return_1m",
    "iv_change",
    "spread_pct",
    "volume_oi_ratio",
    "moneyness",
]

RESULT_COLUMNS = [
    "snapshot_time",
    "underlying",
    "contract_symbol",
    "option_type",
    "expiration",
    "strike",
    "underlying_price",
    "last_price",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "implied_volatility",
    "volume_delta",
    "volume_zscore",
    "cross_section_volume_zscore",
    "volume_oi_ratio",
    "activity_score",
    "anomaly",
    "anomaly_score",
    "method",
    "updated_at",
]


def _warmup_results(frame: pd.DataFrame) -> pd.DataFrame:
    latest_time = frame["snapshot_time"].max()
    latest = frame[frame["snapshot_time"] == latest_time].copy()
    latest["anomaly_score"] = latest["activity_score"].fillna(0)
    cutoff = max(1, int(np.ceil(len(latest) * 0.05)))
    ranked = latest["anomaly_score"].rank(method="first", ascending=False)
    latest["anomaly"] = np.where(ranked <= cutoff, -1, 1)
    latest["method"] = "cross_sectional_warmup"
    latest["updated_at"] = datetime.now(timezone.utc).isoformat()
    return latest[RESULT_COLUMNS]


def detect_spikes(symbol: str, minimum_rows: int = 100) -> dict[str, object]:
    symbol = normalize_symbol(symbol)
    frame = query_frame(
        "SELECT * FROM option_features WHERE underlying=? ORDER BY snapshot_time",
        (symbol,),
    ).replace([np.inf, -np.inf], np.nan)
    if frame.empty:
        raise ValueError(f"No engineered option features exist for {symbol}.")
    usable = frame.dropna(subset=FEATURES).copy()

    if len(usable) < minimum_rows:
        results = _warmup_results(frame)
        method = "cross_sectional_warmup"
        status = "warming_up"
    else:
        model = IsolationForest(
            n_estimators=350,
            contamination=0.01,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(usable[FEATURES])
        usable["anomaly"] = model.predict(usable[FEATURES])
        usable["anomaly_score"] = -model.decision_function(usable[FEATURES])
        usable["method"] = "isolation_forest"
        usable["updated_at"] = datetime.now(timezone.utc).isoformat()
        results = usable.tail(5000)[RESULT_COLUMNS]
        path = artifact_path(symbol, "isolation_forest")
        metrics = {"anomaly_rate": float((usable["anomaly"] == -1).mean())}
        joblib.dump(
            {
                "model": model,
                "features": FEATURES,
                "symbol": symbol,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "training_rows": len(usable),
            },
            path,
        )
        upsert_model_registry(
            symbol, "isolation_forest", str(path), FEATURES, metrics, len(usable)
        )
        method = "isolation_forest"
        status = "ready"

    replace_symbol_frame("anomaly_results", symbol, results)
    latest_time = results["snapshot_time"].max()
    current = results[results["snapshot_time"] == latest_time]
    return {
        "status": status,
        "symbol": symbol,
        "method": method,
        "rows_scored": len(results),
        "current_spikes": int((current["anomaly"] == -1).sum()),
        "history_rows_required": max(0, minimum_rows - len(usable)),
    }
