"""XGBoost production training entry used by the analysis service."""

from models.common import train_classifier


def train_xgboost(symbol: str) -> dict[str, object]:
    return train_classifier(symbol, "xgboost")
