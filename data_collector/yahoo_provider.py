"""Collect live/delayed stock bars and option-chain snapshots with yfinance."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TypeVar

import pandas as pd
import yfinance as yf

from database.db import (
    initialize_database,
    register_symbol,
    save_option_snapshots,
    save_stock_bars,
)

LOGGER = logging.getLogger(__name__)
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-=^]{0,14}$")
T = TypeVar("T")


class MarketDataError(RuntimeError):
    """Raised when the external market-data provider cannot return usable data."""


def normalize_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(
            "Ticker must be 1-15 characters and contain only letters, numbers, '.', '-', '=', or '^'."
        )
    return symbol


def _retry(label: str, operation: Callable[[], T], attempts: int = 3) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:  # noqa: BLE001 - yfinance changes its public exception types
            last_error = error
            LOGGER.warning(
                "%s failed (attempt %s/%s): %s", label, attempt, attempts, error
            )
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise MarketDataError(
        f"{label} failed after {attempts} attempts: {last_error}"
    ) from last_error


def fetch_stock_bars(stock: yf.Ticker, symbol: str) -> pd.DataFrame:
    bars = _retry(
        f"stock bars for {symbol}",
        lambda: stock.history(
            period="5d", interval="1m", auto_adjust=False, prepost=False
        ),
    )
    if bars.empty:
        raise MarketDataError(f"Yahoo returned no one-minute stock bars for {symbol}.")

    bars = bars.reset_index()
    bars = bars.rename(
        columns={
            bars.columns[0]: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).map(
        lambda value: value.isoformat()
    )
    bars["symbol"] = symbol
    columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    bars = bars.reindex(columns=columns)
    for column in ["open", "high", "low", "close", "volume"]:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars.dropna(subset=["timestamp", "close"]).drop_duplicates(
        subset=["timestamp", "symbol"], keep="last"
    )


def fetch_option_chain(
    stock: yf.Ticker,
    symbol: str,
    stock_price: float,
    expiration_count: int,
    strike_range_percent: float,
) -> pd.DataFrame:
    def _load_expirations() -> tuple[yf.Ticker, list[str]]:
    candidate = yf.Ticker(symbol)
    values = list(candidate.options)

    if not values:
        raise RuntimeError(
            f"Yahoo returned an empty option-expiration response for {symbol}"
        )

    return candidate, values


try:
    stock, expirations = _retry(
        f"option expirations for {symbol}",
        _load_expirations,
        attempts=4,
    )
except MarketDataError as error:
    raise MarketDataError(
        f"Yahoo options are temporarily unavailable for {symbol}. "
        "Yahoo may be rate-limiting or rejecting the backend cloud address."
    ) from error





    
    snapshot_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lower_strike = stock_price * (1 - strike_range_percent)
    upper_strike = stock_price * (1 + strike_range_percent)
    frames: list[pd.DataFrame] = []

    # Processing is inside this loop, so every requested expiration is retained.
    for expiration in expirations[:expiration_count]:
        chain = _retry(
            f"option chain {symbol} {expiration}",
            lambda expiration=expiration: stock.option_chain(expiration),
        )
        for option_type, source in (("CALL", chain.calls), ("PUT", chain.puts)):
            if source.empty:
                continue
            frame = source.copy()
            frame["snapshot_time"] = snapshot_time
            frame["underlying"] = symbol
            frame["underlying_price"] = stock_price
            frame["expiration"] = expiration
            frame["option_type"] = option_type
            frame = frame[frame["strike"].between(lower_strike, upper_strike)]
            frames.append(frame)

    if not frames:
        raise MarketDataError(
            f"No {symbol} contracts were found inside the requested strike range. Increase strike_range_percent."
        )

    options = pd.concat(frames, ignore_index=True).rename(
        columns={
            "contractSymbol": "contract_symbol",
            "lastPrice": "last_price",
            "openInterest": "open_interest",
            "impliedVolatility": "implied_volatility",
            "inTheMoney": "in_the_money",
            "contractSize": "contract_size",
        }
    )
    wanted = [
        "snapshot_time",
        "underlying",
        "underlying_price",
        "expiration",
        "option_type",
        "contract_symbol",
        "strike",
        "last_price",
        "bid",
        "ask",
        "volume",
        "open_interest",
        "implied_volatility",
        "in_the_money",
        "contract_size",
        "currency",
    ]
    options = options.reindex(columns=wanted)
    numeric = [
        "underlying_price",
        "strike",
        "last_price",
        "bid",
        "ask",
        "volume",
        "open_interest",
        "implied_volatility",
    ]
    for column in numeric:
        options[column] = pd.to_numeric(options[column], errors="coerce")
    options["in_the_money"] = (
        options["in_the_money"].fillna(False).astype(bool).astype(int)
    )
    return options.dropna(subset=["contract_symbol", "strike"]).drop_duplicates(
        subset=["snapshot_time", "contract_symbol"], keep="last"
    )


def collect_symbol(
    raw_symbol: str,
    expiration_count: int = 2,
    strike_range_percent: float = 0.10,
) -> dict[str, object]:
    symbol = normalize_symbol(raw_symbol)
    expiration_count = max(1, min(5, int(expiration_count)))
    strike_range_percent = max(0.01, min(0.50, float(strike_range_percent)))
    initialize_database()
    register_symbol(symbol)

    stock = yf.Ticker(symbol)
    bars = fetch_stock_bars(stock, symbol)
    stock_price = float(bars["close"].dropna().iloc[-1])
    options = fetch_option_chain(
        stock,
        symbol,
        stock_price,
        expiration_count=expiration_count,
        strike_range_percent=strike_range_percent,
    )
    stock_rows = save_stock_bars(bars)
    option_rows = save_option_snapshots(options)
    LOGGER.info("Collected %s: %s options and %s bars", symbol, option_rows, stock_rows)
    return {
        "symbol": symbol,
        "underlying_price": stock_price,
        "snapshot_time": str(options["snapshot_time"].iloc[0]),
        "option_rows": option_rows,
        "stock_rows": stock_rows,
        "expiration_count": int(options["expiration"].nunique()),
    }
