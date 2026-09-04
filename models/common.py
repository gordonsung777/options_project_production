"""Shared, per-symbol supervised-model training and inference."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from config.settings import get_settings
from data_collector.yahoo_provider import normalize_symbol
from database.db import query_frame, upsert_model_registry

CLASSIFIER_FEATURES = [
    "call_volume_delta",
    "put_volume_delta",
    "put_call_volume_ratio",
    "call_iv_mean",
    "put_iv_mean",
    "iv_skew",
    "call_volume_oi_ratio",
    "put_volume_oi_ratio",
    "call_max_volume_zscore",
    "put_max_volume_zscore",
    "stock_return_1m",
    "stock_return_5m",
    "stock_volume_zscore",
]


def artifact_path(symbol: str, model_name: str, suffix: str = ".joblib") -> Path:
    directory = get_settings().model_dir / normalize_symbol(symbol)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{model_name}{suffix}"


def load_training_data(symbol: str) -> pd.DataFrame:
    frame = query_frame(
        "SELECT * FROM market_features WHERE underlying=? ORDER BY snapshot_minute",
        (normalize_symbol(symbol),),
    ).replace([np.inf, -np.inf], np.nan)
    frame = frame[frame["stock_bar_lag_seconds"].between(0, 120, inclusive="both")]
    return frame.dropna(subset=CLASSIFIER_FEATURES + ["target_up_15m"]).reset_index(
        drop=True
    )


def _metrics(
    y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | None]:
    roc_auc = (
        float(roc_auc_score(y_true, probabilities)) if y_true.nunique() > 1 else None
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": roc_auc,
    }


def train_classifier(
    symbol: str,
    model_name: Literal["random_forest", "xgboost"],
    minimum_rows: int = 120,
) -> dict[str, object]:
    symbol = normalize_symbol(symbol)
    frame = load_training_data(symbol)
    if len(frame) < minimum_rows:
        return {
            "status": "warming_up",
            "model": model_name,
            "rows": len(frame),
            "required_rows": minimum_rows,
            "message": "More labeled minute snapshots are required before supervised training.",
        }

    split = int(len(frame) * 0.80)
    train, test = frame.iloc[:split], frame.iloc[split:]
    y_train = train["target_up_15m"].astype(int)
    y_test = test["target_up_15m"].astype(int)
    if y_train.nunique() < 2 or test.empty:
        return {
            "status": "warming_up",
            "model": model_name,
            "rows": len(frame),
            "required_rows": minimum_rows,
            "message": "Training data needs both UP and NOT_UP target classes.",
        }

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=350,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        negatives = max(1, int((y_train == 0).sum()))
        positives = max(1, int((y_train == 1).sum()))
        model = XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.80,
            colsample_bytree=0.80,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negatives / positives,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(train[CLASSIFIER_FEATURES], y_train)
    probabilities = model.predict_proba(test[CLASSIFIER_FEATURES])[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    metrics = _metrics(y_test, predictions, probabilities)
    baseline_class = int(y_train.value_counts().idxmax())
    metrics["baseline_accuracy"] = float(
        accuracy_score(y_test, np.full(len(y_test), baseline_class))
    )
    path = artifact_path(symbol, model_name)
    bundle = {
        "model": model,
        "features": CLASSIFIER_FEATURES,
        "metrics": metrics,
        "symbol": symbol,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(train),
    }
    joblib.dump(bundle, path)
    upsert_model_registry(
        symbol, model_name, str(path), CLASSIFIER_FEATURES, metrics, len(train)
    )
    return {
        "status": "trained",
        "model": model_name,
        "rows": len(frame),
        "metrics": metrics,
    }


def train_if_stale(
    symbol: str, model_name: Literal["random_forest", "xgboost"]
) -> dict[str, object]:
    symbol = normalize_symbol(symbol)
    current = load_training_data(symbol)
    registry = query_frame(
        "SELECT * FROM model_registry WHERE symbol=? AND model_name=?",
        (symbol, model_name),
    )
    if not registry.empty:
        trained_rows = int(registry.iloc[0]["training_rows"])
        trained_at = pd.to_datetime(registry.iloc[0]["trained_at"], utc=True)
        age_hours = (pd.Timestamp.now(tz="UTC") - trained_at).total_seconds() / 3600
        if len(current) < trained_rows + 20 and age_hours < 24:
            return {"status": "current", "model": model_name, "rows": len(current)}
    return train_classifier(symbol, model_name)


def predict_latest(symbol: str, preferred_model: str = "xgboost") -> dict[str, object]:
    symbol = normalize_symbol(symbol)
    path = artifact_path(symbol, preferred_model)
    if not path.exists():
        return {
            "status": "warming_up",
            "symbol": symbol,
            "model": preferred_model,
            "message": "No trained model exists yet. Repeated scheduled snapshots are required.",
        }
    bundle = joblib.load(path)
    features = list(bundle["features"])
    frame = query_frame(
        "SELECT * FROM market_features WHERE underlying=? ORDER BY snapshot_minute DESC LIMIT 200",
        (symbol,),
    ).replace([np.inf, -np.inf], np.nan)
    complete = frame.dropna(subset=features)
    complete = complete[
        complete["stock_bar_lag_seconds"].between(0, 120, inclusive="both")
    ]
    if complete.empty:
        return {
            "status": "warming_up",
            "symbol": symbol,
            "model": preferred_model,
            "message": "No recent row contains all model features.",
        }
    latest = complete.iloc[[0]]
    probability = float(bundle["model"].predict_proba(latest[features])[0, 1])
    return {
        "status": "ready",
        "symbol": symbol,
        "model": preferred_model,
        "prediction": "UP" if probability >= 0.50 else "NOT_UP",
        "bullish_score": probability,
        "horizon": "15 minutes",
        "data_timestamp": str(latest.iloc[0]["snapshot_minute"]),
        "trained_at": bundle.get("trained_at"),
        "metrics": bundle.get("metrics", {}),
    }


def registry_for_symbol(symbol: str) -> list[dict[str, object]]:
    frame = query_frame(
        "SELECT * FROM model_registry WHERE symbol=? ORDER BY model_name",
        (normalize_symbol(symbol),),
    )
    records: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        record["feature_names"] = json.loads(record.pop("feature_names_json"))
        record["metrics"] = json.loads(record.pop("metrics_json"))
        records.append(record)
    return records
