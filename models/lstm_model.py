"""Optional per-symbol LSTM trainer.

TensorFlow is imported only when this function is called, keeping the normal API
and scheduler image lighter. Install requirements-lstm.txt and set ENABLE_LSTM=true
to activate it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from data_collector.yahoo_provider import normalize_symbol
from database.db import query_frame, upsert_model_registry
from models.common import CLASSIFIER_FEATURES, artifact_path, load_training_data

SEQUENCE_LENGTH = 30


def _sequences(
    features: np.ndarray, targets: np.ndarray, timestamps: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    indices: list[int] = []
    for index in range(SEQUENCE_LENGTH - 1, len(features)):
        start = index - SEQUENCE_LENGTH + 1
        times = timestamps.iloc[start : index + 1]
        gaps = times.diff().dropna().dt.total_seconds() / 60
        if times.dt.date.nunique() != 1 or (not gaps.empty and gaps.max() > 2.5):
            continue
        windows.append(features[start : index + 1])
        labels.append(int(targets[index]))
        indices.append(index)
    return np.asarray(windows), np.asarray(labels), np.asarray(indices)


def train_lstm(symbol: str, minimum_rows: int = 600) -> dict[str, object]:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is optional. Install requirements-lstm.txt before enabling the LSTM."
        ) from error

    symbol = normalize_symbol(symbol)
    frame = load_training_data(symbol)
    if len(frame) < minimum_rows:
        return {
            "status": "warming_up",
            "model": "lstm",
            "rows": len(frame),
            "required_rows": minimum_rows,
        }
    timestamps = pd.to_datetime(frame["snapshot_minute"], utc=True)
    values = frame[CLASSIFIER_FEATURES].to_numpy(dtype=float)
    targets = frame["target_up_15m"].astype(int).to_numpy()
    split = int(len(values) * 0.80)
    scaler = StandardScaler().fit(values[:split])
    windows, labels, target_indices = _sequences(
        scaler.transform(values), targets, timestamps
    )
    train_mask, test_mask = target_indices < split, target_indices >= split
    x_train, y_train = windows[train_mask], labels[train_mask]
    x_test, y_test = windows[test_mask], labels[test_mask]
    if len(x_train) < 100 or len(x_test) < 20 or len(np.unique(y_train)) < 2:
        return {
            "status": "warming_up",
            "model": "lstm",
            "rows": len(frame),
            "message": "Not enough contiguous sequences or target-class variety.",
        }

    validation_index = int(len(x_train) * 0.80)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((SEQUENCE_LENGTH, len(CLASSIFIER_FEATURES))),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        x_train[:validation_index],
        y_train[:validation_index],
        validation_data=(x_train[validation_index:], y_train[validation_index:]),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
        shuffle=False,
        verbose=0,
    )
    probabilities = model.predict(x_test, verbose=0).ravel()
    predictions = (probabilities >= 0.50).astype(int)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities))
        if len(np.unique(y_test)) > 1
        else None,
    }
    model_path = artifact_path(symbol, "lstm", ".keras")
    scaler_path = artifact_path(symbol, "lstm_scaler")
    model.save(model_path)
    joblib.dump(
        {
            "scaler": scaler,
            "features": CLASSIFIER_FEATURES,
            "sequence_length": SEQUENCE_LENGTH,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        scaler_path,
    )
    upsert_model_registry(
        symbol, "lstm", str(model_path), CLASSIFIER_FEATURES, metrics, len(x_train)
    )
    return {
        "status": "trained",
        "model": "lstm",
        "rows": len(frame),
        "metrics": metrics,
    }


def train_lstm_if_stale(symbol: str) -> dict[str, object]:
    symbol = normalize_symbol(symbol)
    current = load_training_data(symbol)
    registry = query_frame(
        "SELECT * FROM model_registry WHERE symbol=? AND model_name='lstm'", (symbol,)
    )
    if not registry.empty:
        trained_rows = int(registry.iloc[0]["training_rows"])
        trained_at = pd.to_datetime(registry.iloc[0]["trained_at"], utc=True)
        age_hours = (pd.Timestamp.now(tz="UTC") - trained_at).total_seconds() / 3600
        if len(current) < trained_rows + 60 and age_hours < 24:
            return {"status": "current", "model": "lstm", "rows": len(current)}
    return train_lstm(symbol)
