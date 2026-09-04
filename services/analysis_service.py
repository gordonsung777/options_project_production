"""Single production workflow used by FastAPI and the background scheduler."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_settings
from data_collector.yahoo_provider import collect_symbol, normalize_symbol
from database.db import (
    finish_pipeline_run,
    get_tracked_symbols,
    initialize_database,
    query_frame,
    start_pipeline_run,
)
from features.feature_engineering import create_model_dataset
from models.common import predict_latest, registry_for_symbol, train_if_stale
from models.isolation_forest import detect_spikes
from models.lstm_model import train_lstm_if_stale

LOGGER = logging.getLogger(__name__)
_LOCKS: dict[str, Lock] = {}
_LOCKS_GUARD = Lock()


def _symbol_lock(symbol: str) -> Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(symbol, Lock())


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.replace([np.inf, -np.inf], np.nan).astype(object)
    clean = clean.where(pd.notnull(clean), None)
    return clean.to_dict(orient="records")


def analyze_symbol(
    raw_symbol: str,
    expiration_count: int | None = None,
    strike_range_percent: float | None = None,
    train_models: bool = True,
) -> dict[str, object]:
    settings = get_settings()
    symbol = normalize_symbol(raw_symbol)
    expiration_count = expiration_count or settings.expiration_count
    strike_range_percent = strike_range_percent or settings.strike_range_percent
    initialize_database()
    run_id = start_pipeline_run(symbol)

    with _symbol_lock(symbol):
        try:
            collection = collect_symbol(symbol, expiration_count, strike_range_percent)
            features = create_model_dataset(symbol)
            spikes = detect_spikes(symbol)
            models: list[dict[str, object]] = []
            if train_models and settings.auto_train:
                models.append(train_if_stale(symbol, "random_forest"))
                models.append(train_if_stale(symbol, "xgboost"))
                if settings.enable_lstm:
                    models.append(train_lstm_if_stale(symbol))
            prediction = predict_latest(symbol)
            finish_pipeline_run(
                run_id,
                "succeeded",
                "Live collection and analysis completed.",
                int(collection["option_rows"]),
                int(collection["stock_rows"]),
            )
            return {
                "status": "succeeded",
                "run_id": run_id,
                "collection": collection,
                "features": features,
                "spike_detection": spikes,
                "model_training": models,
                "prediction": prediction,
            }
        except Exception as error:
            finish_pipeline_run(run_id, "failed", f"{type(error).__name__}: {error}")
            LOGGER.exception("Analysis failed for %s", symbol)
            raise


def analysis_payload(raw_symbol: str, history_limit: int = 390) -> dict[str, object]:
    symbol = normalize_symbol(raw_symbol)
    history_limit = max(20, min(2000, int(history_limit)))
    market = query_frame(
        """
        SELECT * FROM (
            SELECT * FROM market_features
            WHERE underlying=?
            ORDER BY snapshot_minute DESC
            LIMIT ?
        ) ORDER BY snapshot_minute
        """,
        (symbol, history_limit),
    )
    latest_options = query_frame(
        """
        SELECT * FROM option_snapshots
        WHERE underlying=? AND snapshot_time=(
            SELECT MAX(snapshot_time) FROM option_snapshots WHERE underlying=?
        )
        ORDER BY volume DESC LIMIT 100
        """,
        (symbol, symbol),
    )
    spikes = spike_payload(symbol, 50)
    latest_market = _records(market.tail(1))[0] if not market.empty else None
    return {
        "symbol": symbol,
        "latest": latest_market,
        "history": _records(market),
        "latest_contracts": _records(latest_options),
        "spikes": spikes,
        "prediction": predict_latest(symbol),
        "models": registry_for_symbol(symbol),
    }


def spike_payload(raw_symbol: str, limit: int = 50) -> dict[str, object]:
    symbol = normalize_symbol(raw_symbol)
    limit = max(1, min(200, int(limit)))
    frame = query_frame(
        """
        SELECT * FROM anomaly_results
        WHERE underlying=? AND snapshot_time=(
            SELECT MAX(snapshot_time) FROM anomaly_results WHERE underlying=?
        )
        ORDER BY anomaly ASC, anomaly_score DESC
        LIMIT ?
        """,
        (symbol, symbol, limit),
    )
    method = str(frame.iloc[0]["method"]) if not frame.empty else None
    timestamp = str(frame.iloc[0]["snapshot_time"]) if not frame.empty else None
    return {
        "symbol": symbol,
        "method": method,
        "data_timestamp": timestamp,
        "results": _records(frame),
    }


def symbols_payload() -> list[dict[str, object]]:
    symbols = get_tracked_symbols()
    if not symbols:
        return []
    placeholders = ",".join("?" for _ in symbols)
    frame = query_frame(
        f"""
        SELECT underlying AS symbol, MAX(snapshot_time) AS latest_snapshot, COUNT(*) AS snapshot_rows
        FROM option_snapshots
        WHERE underlying IN ({placeholders})
        GROUP BY underlying ORDER BY underlying
        """,
        symbols,
    )
    counts = {row["symbol"]: row for row in _records(frame)}
    return [
        counts.get(
            symbol, {"symbol": symbol, "latest_snapshot": None, "snapshot_rows": 0}
        )
        for symbol in symbols
    ]
