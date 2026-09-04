"""Production FastAPI entry point."""

from __future__ import annotations

import hmac
import logging
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AnalyzeRequest
from config.settings import get_settings
from data_collector.yahoo_provider import MarketDataError, normalize_symbol
from database.db import get_connection, initialize_database
from models.common import predict_latest
from services.analysis_service import (
    analysis_payload,
    analyze_symbol,
    spike_payload,
    symbols_payload,
)

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_database()
    yield


app = FastAPI(
    title="Options Market Intelligence API",
    description="On-demand options activity, anomaly detection, and 15-minute model inference.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if settings.api_key and (
        x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key)
    ):
        raise HTTPException(
            status_code=401, detail="A valid X-API-Key header is required."
        )


protected = [Depends(require_api_key)]


@app.get("/", tags=["service"])
def root() -> dict[str, object]:
    return {
        "service": "Options Market Intelligence API",
        "version": "2.0.0",
        "docs": "/docs",
        "workflow": "POST /v1/analyze, then GET /v1/analysis/{symbol}",
    }


@app.get("/health", tags=["service"])
def health() -> dict[str, object]:
    try:
        with get_connection() as connection:
            database_ok = connection.execute("SELECT 1").fetchone()[0] == 1
            tracked = connection.execute(
                "SELECT COUNT(*) FROM tracked_symbols"
            ).fetchone()[0]
        return {
            "status": "healthy",
            "database": database_ok,
            "tracked_symbols": tracked,
        }
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=503, detail=f"Database unavailable: {error}"
        ) from error


@app.post("/v1/analyze", dependencies=protected, tags=["analysis"])
def analyze(request: AnalyzeRequest) -> dict[str, object]:
    try:
        return analyze_symbol(
            request.symbol,
            request.expiration_count,
            request.strike_range_percent,
            request.train_models,
        )
    except MarketDataError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=503, detail=f"Database operation failed: {error}"
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {error}"
        ) from error


@app.get("/v1/analysis/{symbol}", dependencies=protected, tags=["analysis"])
def get_analysis(
    symbol: str, history_limit: int = Query(default=390, ge=20, le=2000)
) -> dict[str, object]:
    try:
        return analysis_payload(normalize_symbol(symbol), history_limit)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/v1/spikes/{symbol}", dependencies=protected, tags=["analysis"])
def get_spikes(
    symbol: str, limit: int = Query(default=50, ge=1, le=200)
) -> dict[str, object]:
    try:
        return spike_payload(normalize_symbol(symbol), limit)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/v1/predict/{symbol}", dependencies=protected, tags=["models"])
def get_prediction(symbol: str) -> dict[str, object]:
    try:
        return predict_latest(normalize_symbol(symbol))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/v1/symbols", dependencies=protected, tags=["analysis"])
def get_symbols() -> list[dict[str, object]]:
    return symbols_payload()
