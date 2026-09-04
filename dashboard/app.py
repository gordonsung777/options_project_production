"""Streamlit client for the production FastAPI service."""

from __future__ import annotations

import os
import re
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY")
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-=^]{0,14}$")

st.set_page_config(
    page_title="Options Spike Intelligence", page_icon="📈", layout="wide"
)
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background:#111827; border:1px solid #273449; padding:1rem; border-radius:14px;}
      div.stButton > button {border-radius:12px; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_request(method: str, path: str, **kwargs: Any) -> Any:
    headers = dict(kwargs.pop("headers", {}))
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    with httpx.Client(timeout=150.0) as client:
        response = client.request(method, f"{API_URL}{path}", headers=headers, **kwargs)
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API {response.status_code}: {detail}")
    return response.json()


def number(value: Any, digits: int = 2, prefix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{prefix}{float(value):,.{digits}f}"


st.title("📈 Options Spike Intelligence")
st.caption(
    "Enter a ticker, collect the latest option chain, and inspect unusual activity."
)

with st.sidebar:
    st.header("Analyze a stock")
    typed_symbol = st.text_input(
        "Ticker symbol", value=st.session_state.get("symbol", "SPY"), max_chars=15
    )
    expiration_count = st.slider("Expirations", 1, 5, 2)
    strike_percent = st.slider("Strike range around price", 1, 50, 10) / 100
    train_models = st.toggle("Train models when ready", value=True)
    analyze_clicked = st.button(
        "Run live analysis", type="primary", use_container_width=True
    )
    refresh_clicked = st.button("Refresh saved results", use_container_width=True)
    st.divider()
    st.caption(
        "Analytics only—not investment advice. Yahoo data can be delayed or incomplete."
    )

symbol = typed_symbol.strip().upper()
if not SYMBOL_PATTERN.fullmatch(symbol):
    st.error("Enter a valid ticker such as AAPL, NVDA, SPY, or BRK-B.")
    st.stop()
st.session_state["symbol"] = symbol

if analyze_clicked:
    with st.status(f"Collecting and analyzing {symbol}…", expanded=True) as status:
        try:
            run = api_request(
                "POST",
                "/v1/analyze",
                json={
                    "symbol": symbol,
                    "expiration_count": expiration_count,
                    "strike_range_percent": strike_percent,
                    "train_models": train_models,
                },
            )
            st.write(f"Saved {run['collection']['option_rows']:,} option contracts.")
            st.write(
                f"Built {run['features']['option_feature_rows']:,} option feature rows."
            )
            status.update(
                label=f"{symbol} analysis completed", state="complete", expanded=False
            )
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            status.update(label=f"{symbol} analysis failed", state="error")
            st.error(str(error))

if refresh_clicked:
    st.rerun()

try:
    payload = api_request(
        "GET", f"/v1/analysis/{symbol}", params={"history_limit": 780}
    )
except (httpx.HTTPError, RuntimeError, ValueError) as error:
    st.info(
        f"No saved analysis is available for {symbol}. Press **Run live analysis**."
    )
    st.caption(str(error))
    st.stop()

latest = payload.get("latest")
contracts = pd.DataFrame(payload.get("latest_contracts", []))
history = pd.DataFrame(payload.get("history", []))
spikes_payload = payload.get("spikes", {})
spikes = pd.DataFrame(spikes_payload.get("results", []))
prediction = payload.get("prediction", {})

if not latest and contracts.empty:
    st.info(
        f"No saved analysis is available for {symbol}. Press **Run live analysis**."
    )
    st.stop()

latest_price = (
    latest.get("stock_close")
    if latest
    else contracts.get("underlying_price", pd.Series([None])).iloc[0]
)
metric_columns = st.columns(5)
metric_columns[0].metric("Underlying", symbol)
metric_columns[1].metric("Last price", number(latest_price, 2, "$"))
metric_columns[2].metric(
    "Call new volume", number(latest.get("call_volume_delta") if latest else None, 0)
)
metric_columns[3].metric(
    "Put new volume", number(latest.get("put_volume_delta") if latest else None, 0)
)
metric_columns[4].metric(
    "Put / call", number(latest.get("put_call_volume_ratio") if latest else None, 2)
)

method = spikes_payload.get("method")
if method == "cross_sectional_warmup":
    st.warning(
        "Warm-up mode: these are contracts unusual relative to the current chain. "
        "The scheduler must collect repeated snapshots before Isolation Forest can identify time-series spikes."
    )
elif method == "isolation_forest":
    st.success(
        "Isolation Forest is scoring historical changes in volume, price, IV, spread, and moneyness."
    )

overview_tab, spikes_tab, contracts_tab, model_tab = st.tabs(
    ["Market overview", "Spike detector", "Contract explorer", "Model status"]
)

with overview_tab:
    if not history.empty:
        history["snapshot_minute"] = pd.to_datetime(
            history["snapshot_minute"], utc=True
        )
        left, right = st.columns(2)
        with left:
            price_figure = px.line(
                history,
                x="snapshot_minute",
                y="stock_close",
                title=f"{symbol} underlying price",
            )
            price_figure.update_layout(
                height=360, yaxis_title="Price", xaxis_title=None
            )
            st.plotly_chart(price_figure, use_container_width=True)
        with right:
            activity_figure = go.Figure()
            activity_figure.add_trace(
                go.Bar(
                    x=history["snapshot_minute"],
                    y=history["call_volume_delta"],
                    name="Calls",
                )
            )
            activity_figure.add_trace(
                go.Bar(
                    x=history["snapshot_minute"],
                    y=history["put_volume_delta"],
                    name="Puts",
                )
            )
            activity_figure.update_layout(
                title="Estimated new option volume",
                barmode="group",
                height=360,
                xaxis_title=None,
            )
            st.plotly_chart(activity_figure, use_container_width=True)
    else:
        st.info(
            "Option history exists, but a matching stock bar is not yet available for the overview chart."
        )

with spikes_tab:
    if spikes.empty:
        st.info("No spike scores are saved yet.")
    else:
        show_all = st.toggle("Show normal contracts too", value=False)
        displayed = spikes if show_all else spikes[spikes["anomaly"] == -1]
        st.dataframe(
            displayed[
                [
                    "contract_symbol",
                    "option_type",
                    "expiration",
                    "strike",
                    "volume",
                    "open_interest",
                    "volume_delta",
                    "volume_oi_ratio",
                    "anomaly_score",
                    "method",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

with contracts_tab:
    if contracts.empty:
        st.info("No current contracts are saved.")
    else:
        option_type = st.segmented_control(
            "Option type", ["ALL", "CALL", "PUT"], default="ALL"
        )
        filtered = (
            contracts
            if option_type == "ALL"
            else contracts[contracts["option_type"] == option_type]
        )
        st.dataframe(
            filtered[
                [
                    "contract_symbol",
                    "option_type",
                    "expiration",
                    "strike",
                    "last_price",
                    "bid",
                    "ask",
                    "volume",
                    "open_interest",
                    "implied_volatility",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

with model_tab:
    if prediction.get("status") == "ready":
        col1, col2 = st.columns(2)
        col1.metric("15-minute classification", prediction["prediction"])
        col2.metric("Bullish model score", f"{prediction['bullish_score']:.1%}")
        st.progress(float(prediction["bullish_score"]))
        st.caption(
            "A model score is not a guaranteed probability and is not a trade instruction."
        )
    else:
        st.info(prediction.get("message", "The predictive model is warming up."))
    models = payload.get("models", [])
    if models:
        rows = []
        for item in models:
            metrics = item.get("metrics", {})
            rows.append(
                {
                    "model": item["model_name"],
                    "trained_at": item["trained_at"],
                    "training_rows": item["training_rows"],
                    "accuracy": metrics.get("accuracy"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
