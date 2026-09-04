"""Environment-driven settings shared by the API, dashboard, and scheduler."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(
        item.strip() for item in os.getenv(name, default).split(",") if item.strip()
    )


def _boolean(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    db_path: Path
    model_dir: Path
    default_symbols: tuple[str, ...]
    expiration_count: int
    strike_range_percent: float
    collection_interval_seconds: int
    closed_market_sleep_seconds: int
    auto_train: bool
    enable_lstm: bool
    api_key: str | None
    cors_origins: tuple[str, ...]
    api_url: str
    log_level: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        db_path=Path(
            os.getenv("OPTIONS_DB_PATH", PROJECT_ROOT / "runtime" / "options_market.db")
        ),
        model_dir=Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "runtime" / "models")),
        default_symbols=tuple(
            symbol.upper() for symbol in _csv("DEFAULT_SYMBOLS", "SPY,QQQ,NVDA")
        ),
        expiration_count=max(1, min(5, int(os.getenv("EXPIRATION_COUNT", "2")))),
        strike_range_percent=max(
            0.01, min(0.50, float(os.getenv("STRIKE_RANGE_PERCENT", "0.10")))
        ),
        collection_interval_seconds=max(
            60, int(os.getenv("COLLECTION_INTERVAL_SECONDS", "60"))
        ),
        closed_market_sleep_seconds=max(
            60, int(os.getenv("CLOSED_MARKET_SLEEP_SECONDS", "300"))
        ),
        auto_train=_boolean("AUTO_TRAIN", True),
        enable_lstm=_boolean("ENABLE_LSTM", False),
        api_key=os.getenv("API_KEY") or None,
        cors_origins=_csv("CORS_ORIGINS", "http://localhost:8501"),
        api_url=os.getenv("API_URL", "http://localhost:8000").rstrip("/"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
