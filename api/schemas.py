"""Validated public API request models."""

from pydantic import BaseModel, Field, field_validator

from data_collector.yahoo_provider import normalize_symbol


class AnalyzeRequest(BaseModel):
    symbol: str = Field(examples=["AAPL"], description="Ticker entered by the client")
    expiration_count: int = Field(default=2, ge=1, le=5)
    strike_range_percent: float = Field(default=0.10, ge=0.01, le=0.50)
    train_models: bool = True

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        return normalize_symbol(value)
