"""Build leakage-aware option, stock, and model features for one ticker."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_collector.yahoo_provider import normalize_symbol
from database.db import query_frame, replace_symbol_frame

OPTION_FEATURE_COLUMNS = [
    "snapshot_time",
    "snapshot_minute",
    "trade_date",
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
    "mid_price",
    "spread_pct",
    "moneyness",
    "days_to_expiration",
    "snapshot_gap_minutes",
    "volume_delta",
    "option_return_1m",
    "iv_change",
    "volume_oi_ratio",
    "volume_mean_20",
    "volume_std_20",
    "volume_zscore",
    "cross_section_volume_zscore",
    "activity_score",
]

MARKET_FEATURE_COLUMNS = [
    "underlying",
    "snapshot_minute",
    "call_volume_delta",
    "put_volume_delta",
    "call_total_volume",
    "put_total_volume",
    "call_open_interest",
    "put_open_interest",
    "call_iv_mean",
    "put_iv_mean",
    "call_max_volume_zscore",
    "put_max_volume_zscore",
    "call_max_activity_score",
    "put_max_activity_score",
    "put_call_volume_ratio",
    "call_volume_oi_ratio",
    "put_volume_oi_ratio",
    "iv_skew",
    "stock_bar_minute",
    "stock_close",
    "stock_volume",
    "stock_return_1m",
    "stock_return_5m",
    "stock_volume_zscore",
    "future_return_15m",
    "target_up_15m",
    "stock_bar_lag_seconds",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _robust_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(0.0, index=series.index)
    return (values - median) / (1.4826 * mad)


def build_option_features(symbol: str) -> pd.DataFrame:
    symbol = normalize_symbol(symbol)
    frame = query_frame(
        "SELECT * FROM option_snapshots WHERE underlying=? ORDER BY snapshot_time, contract_symbol",
        (symbol,),
    )
    if frame.empty:
        raise ValueError(f"No option snapshots exist for {symbol}.")

    frame["snapshot_time"] = pd.to_datetime(frame["snapshot_time"], utc=True)
    frame["snapshot_minute"] = frame["snapshot_time"].dt.floor("min")
    frame["trade_date"] = frame["snapshot_time"].dt.date.astype(str)
    frame["expiration"] = pd.to_datetime(frame["expiration"], errors="coerce")
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
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.sort_values("snapshot_time")
        .drop_duplicates(["contract_symbol", "snapshot_minute"], keep="last")
        .sort_values(["contract_symbol", "snapshot_minute"])
        .reset_index(drop=True)
    )

    valid_quote = (frame["bid"] > 0) & (frame["ask"] >= frame["bid"])
    frame["mid_price"] = np.where(
        valid_quote, (frame["bid"] + frame["ask"]) / 2, frame["last_price"]
    )
    frame["spread_pct"] = safe_divide(frame["ask"] - frame["bid"], frame["mid_price"])
    frame["moneyness"] = safe_divide(
        frame["strike"] - frame["underlying_price"], frame["underlying_price"]
    )
    snapshot_date = frame["snapshot_time"].dt.tz_localize(None).dt.normalize()
    frame["days_to_expiration"] = (frame["expiration"] - snapshot_date).dt.days

    groups = frame.groupby(["contract_symbol", "trade_date"], sort=False)
    frame["snapshot_gap_minutes"] = (
        groups["snapshot_minute"].diff().dt.total_seconds() / 60
    )
    frame["volume_delta"] = groups["volume"].diff()
    frame.loc[frame["volume_delta"] < 0, "volume_delta"] = np.nan
    bad_gap = frame["snapshot_gap_minutes"] > 2.5
    frame.loc[bad_gap, "volume_delta"] = np.nan
    frame["option_return_1m"] = groups["mid_price"].pct_change(fill_method=None)
    frame["iv_change"] = groups["implied_volatility"].diff()
    frame.loc[bad_gap, ["option_return_1m", "iv_change"]] = np.nan
    frame["volume_oi_ratio"] = safe_divide(frame["volume"], frame["open_interest"])

    groups = frame.groupby(["contract_symbol", "trade_date"], sort=False)
    frame["volume_mean_20"] = groups["volume_delta"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=5).mean()
    )
    frame["volume_std_20"] = groups["volume_delta"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=5).std()
    )
    frame["volume_zscore"] = safe_divide(
        frame["volume_delta"] - frame["volume_mean_20"], frame["volume_std_20"]
    )

    log_volume = np.log1p(frame["volume"].clip(lower=0).fillna(0))
    frame["cross_section_volume_zscore"] = log_volume.groupby(
        [frame["snapshot_minute"], frame["option_type"]]
    ).transform(_robust_zscore)
    temporal = frame["volume_zscore"].fillna(0).clip(lower=0, upper=10)
    cross_section = (
        frame["cross_section_volume_zscore"].fillna(0).clip(lower=0, upper=10)
    )
    volume_oi = np.log1p(frame["volume_oi_ratio"].clip(lower=0, upper=20).fillna(0))
    frame["activity_score"] = 0.50 * temporal + 0.35 * cross_section + 0.15 * volume_oi
    return frame


def aggregate_option_market(option_features: pd.DataFrame) -> pd.DataFrame:
    keys = ["underlying", "snapshot_minute"]

    def aggregate_side(option_type: str, prefix: str) -> pd.DataFrame:
        side = option_features[option_features["option_type"] == option_type]
        if side.empty:
            return pd.DataFrame(columns=keys)
        result = (
            side.groupby(keys)
            .agg(
                volume_delta=("volume_delta", lambda values: values.sum(min_count=1)),
                total_volume=("volume", "sum"),
                open_interest=("open_interest", "sum"),
                iv_mean=("implied_volatility", "mean"),
                max_volume_zscore=("volume_zscore", "max"),
                max_activity_score=("activity_score", "max"),
            )
            .reset_index()
        )
        return result.rename(
            columns={
                column: f"{prefix}_{column}"
                for column in result.columns
                if column not in keys
            }
        )

    market = aggregate_side("CALL", "call").merge(
        aggregate_side("PUT", "put"), on=keys, how="outer"
    )
    for column in (
        "call_volume_delta",
        "put_volume_delta",
        "call_total_volume",
        "put_total_volume",
        "call_open_interest",
        "put_open_interest",
        "call_iv_mean",
        "put_iv_mean",
        "call_max_volume_zscore",
        "put_max_volume_zscore",
        "call_max_activity_score",
        "put_max_activity_score",
    ):
        if column not in market:
            market[column] = np.nan
    market["put_call_volume_ratio"] = safe_divide(
        market["put_volume_delta"] + 1, market["call_volume_delta"] + 1
    )
    market["call_volume_oi_ratio"] = safe_divide(
        market["call_volume_delta"], market["call_open_interest"]
    )
    market["put_volume_oi_ratio"] = safe_divide(
        market["put_volume_delta"], market["put_open_interest"]
    )
    market["iv_skew"] = market["put_iv_mean"] - market["call_iv_mean"]
    return market.sort_values("snapshot_minute")


def build_stock_features(symbol: str) -> pd.DataFrame:
    stock = query_frame(
        "SELECT * FROM stock_bars WHERE symbol=? ORDER BY timestamp",
        (normalize_symbol(symbol),),
    )
    if stock.empty:
        raise ValueError(f"No stock bars exist for {symbol}.")
    stock["timestamp"] = pd.to_datetime(stock["timestamp"], utc=True)
    stock["stock_bar_minute"] = stock["timestamp"].dt.floor("min")
    stock["trade_date"] = stock["timestamp"].dt.date.astype(str)
    for column in ["open", "high", "low", "close", "volume"]:
        stock[column] = pd.to_numeric(stock[column], errors="coerce")
    stock = (
        stock.sort_values("timestamp")
        .drop_duplicates("stock_bar_minute", keep="last")
        .reset_index(drop=True)
    )
    groups = stock.groupby(["symbol", "trade_date"], sort=False)
    stock["stock_return_1m"] = groups["close"].pct_change(fill_method=None)
    stock["stock_return_5m"] = groups["close"].pct_change(5, fill_method=None)
    stock["stock_volume_mean_20"] = groups["volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=5).mean()
    )
    stock["stock_volume_std_20"] = groups["volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=5).std()
    )
    stock["stock_volume_zscore"] = safe_divide(
        stock["volume"] - stock["stock_volume_mean_20"], stock["stock_volume_std_20"]
    )
    stock["future_close_15m"] = groups["close"].shift(-15)
    future_time = groups["stock_bar_minute"].shift(-15)
    gap = (future_time - stock["stock_bar_minute"]).dt.total_seconds() / 60
    stock["future_return_15m"] = stock["future_close_15m"] / stock["close"] - 1
    stock.loc[~gap.between(14, 17), "future_return_15m"] = np.nan
    stock["target_up_15m"] = np.where(
        stock["future_return_15m"].notna(),
        (stock["future_return_15m"] > 0).astype(int),
        np.nan,
    )
    return stock


def create_model_dataset(symbol: str) -> dict[str, object]:
    symbol = normalize_symbol(symbol)
    option_features = build_option_features(symbol)
    option_market = aggregate_option_market(option_features)
    stock = build_stock_features(symbol)
    stock_for_merge = stock[
        [
            "stock_bar_minute",
            "close",
            "volume",
            "stock_return_1m",
            "stock_return_5m",
            "stock_volume_zscore",
            "future_return_15m",
            "target_up_15m",
        ]
    ].rename(columns={"close": "stock_close", "volume": "stock_volume"})

    model_data = pd.merge_asof(
        option_market.sort_values("snapshot_minute"),
        stock_for_merge.sort_values("stock_bar_minute"),
        left_on="snapshot_minute",
        right_on="stock_bar_minute",
        direction="backward",
        # A wider merge keeps after-hours dashboard snapshots visible. The
        # supervised trainer separately rejects rows whose stock bar is stale.
        tolerance=pd.Timedelta(days=4),
    ).dropna(subset=["stock_bar_minute", "stock_close"])
    model_data["stock_bar_lag_seconds"] = (
        model_data["snapshot_minute"] - model_data["stock_bar_minute"]
    ).dt.total_seconds()

    save_options = option_features[OPTION_FEATURE_COLUMNS].copy()
    save_model = model_data[MARKET_FEATURE_COLUMNS].copy()
    for frame, columns in (
        (save_options, ["snapshot_time", "snapshot_minute", "expiration"]),
        (save_model, ["snapshot_minute", "stock_bar_minute"]),
    ):
        for column in columns:
            frame[column] = frame[column].astype(str)
    replace_symbol_frame("option_features", symbol, save_options)
    replace_symbol_frame("market_features", symbol, save_model)
    return {
        "symbol": symbol,
        "option_feature_rows": len(option_features),
        "market_feature_rows": len(model_data),
        "historical_snapshots": int(option_features["snapshot_minute"].nunique()),
    }
