"""Production scheduler that continuously analyzes registered tickers while NYSE is open."""

from __future__ import annotations

import logging
import signal
from datetime import datetime
from threading import Event
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

from config.settings import get_settings
from database.db import get_tracked_symbols, initialize_database, register_symbol
from services.analysis_service import analyze_symbol

LOGGER = logging.getLogger(__name__)
NEW_YORK = ZoneInfo("America/New_York")
NYSE = mcal.get_calendar("NYSE")
STOP = Event()


def market_is_open() -> bool:
    now = pd.Timestamp.now(tz=NEW_YORK)
    schedule = NYSE.schedule(start_date=now.date(), end_date=now.date())
    if schedule.empty:
        return False
    market_open = schedule.iloc[0]["market_open"].tz_convert(NEW_YORK)
    market_close = schedule.iloc[0]["market_close"].tz_convert(NEW_YORK)
    return bool(market_open <= now <= market_close)


def _stop(_: int, __: object) -> None:
    STOP.set()


def run_collection_cycle() -> None:
    settings = get_settings()
    symbols = get_tracked_symbols()
    LOGGER.info("Starting cycle for %s", symbols)
    for symbol in symbols:
        if STOP.is_set():
            return
        try:
            analyze_symbol(
                symbol,
                settings.expiration_count,
                settings.strike_range_percent,
                train_models=True,
            )
        except Exception:
            LOGGER.exception("Scheduled analysis failed for %s", symbol)


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    initialize_database()
    for symbol in settings.default_symbols:
        register_symbol(symbol)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    LOGGER.info("Scheduler started with %s", get_tracked_symbols())
    while not STOP.is_set():
        if market_is_open():
            run_collection_cycle()
            STOP.wait(settings.collection_interval_seconds)
        else:
            LOGGER.info("NYSE closed at %s", datetime.now(NEW_YORK).isoformat())
            STOP.wait(settings.closed_market_sleep_seconds)
    LOGGER.info("Scheduler stopped")


if __name__ == "__main__":
    main()
