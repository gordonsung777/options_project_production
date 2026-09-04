"""Random Forest production training entry used by the analysis service."""

from models.common import train_classifier


def train_random_forest(symbol: str) -> dict[str, object]:
    return train_classifier(symbol, "random_forest")
