# Production Audit Report

## Outcome

The uploaded learning project was rebuilt into a normalized production directory. The original uploads remain unchanged. The corrected design accepts arbitrary validated tickers through both Streamlit and FastAPI, registers those tickers for scheduled collection, isolates features/models per symbol, and removes collector/model demo entry blocks.

## Findings and fixes

| Original file/area | Production problem found | Correction |
|---|---|---|
| `yahoo_options(1).py` | CALL/PUT processing was outside the expiration loop, so only the final fetched expiration was processed. | Processing now occurs inside the loop for every requested expiration. |
| `yahoo_options(1).py` | A hard-coded `NVDA` test block executed direct collection. | Removed; live collection is invoked through the service/API/scheduler. |
| `stock_collector(1).py` | A hard-coded `NVDA` test block and console preview remained. | Removed; the function is a reusable production collector. |
| `db(1).py` | Option snapshots had no primary key, allowing duplicate rows. | Added `(snapshot_time, contract_symbol)` primary key and upserts. |
| `db(1).py` | No dynamic watchlist, model registry, or pipeline-run audit. | Added `tracked_symbols`, `model_registry`, and `pipeline_runs`. |
| `scheduler(1).py` | Symbols were hard-coded; clients could not add their own ticker. | Scheduler reads the database watchlist; any API/dashboard ticker is registered automatically. |
| `scheduler(1).py` | Collection logic differed from API behavior. | Scheduler and API now call the same `analyze_symbol()` workflow. |
| `feature_engineering.py` | Replacing complete feature tables risked deleting other symbols during an update. | Feature writes replace rows only for the requested symbol. |
| `feature_engineering.py` | A 15-row shift could span a missing-data gap. | The future target is accepted only when the time gap is 14–17 minutes. |
| `feature_engineering.py` | No first-snapshot activity ranking existed. | Added robust current-chain scoring while historical features warm up. |
| `isolation_forest.py` | One global model mixed every ticker and overwrote one artifact. | Models and results are per ticker. |
| `random_forest.py` / `xgboost_model.py` | One global chronological split mixed symbols with different market behavior. | Each model trains on only one ticker. |
| Supervised models | Re-trained whenever launched, even with no useful new rows. | Registry-based staleness check waits for 20 new rows or 24 hours. |
| `lstm_model.py` | Training was hard-coded to `SPY`. | LSTM accepts any validated ticker and saves per-symbol artifacts. |
| `lstm_model.py` | TensorFlow was required by the base runtime. | Made optional through `requirements-lstm.txt` and `ENABLE_LSTM`. |
| `main(3).py` | API only read existing tables; it could not analyze a client-entered ticker. | Added `POST /v1/analyze` for collection and full analysis. |
| `main(3).py` | Global XGBoost artifact could be applied to the wrong symbol. | Prediction loads `runtime/models/<SYMBOL>/xgboost.joblib`. |
| `main(3).py` | No request validation, optional authentication, or controlled CORS. | Added Pydantic bounds, ticker validation, optional API key, and CORS configuration. |
| `app(1).py` | Only symbols already in the database appeared in a select box. | Added a free-form ticker input and a live analysis button. |
| `app(1).py` | Streamlit read SQLite and models directly. | Streamlit is now an API client, giving one controlled write path. |
| `app(1).py` | Empty/broad exception handling hid operational failures. | Dashboard displays API status and provider/database error details. |
| `requirements(2).txt` | Dependencies were unpinned and TensorFlow made every deployment heavy. | Pinned the base environment; separated optional LSTM dependencies. |
| File layout | Uploaded names such as `(1)` and `(3)` did not match imports. | Created standard package paths and `__init__.py` files. |
| Docker/runtime | No reliable three-service production handoff was available in the uploaded source set. | Added non-root image, health checks, restart policies, graceful scheduler stop, and persistent volume. |

## Removed test/demo behavior

- Removed hard-coded ticker executions from collectors and models.
- Removed preview `print()` tables used as manual tests.
- Retained only the scheduler’s `__main__` block because it is the real production process invoked by Docker.
- Validation was run externally against temporary runtime data; validation helpers are not shipped as application endpoints or methods.

## Remaining external limitations

- Yahoo/yfinance is not an exchange-grade licensed feed.
- A first request cannot produce a historical spike; it produces an explicitly labeled current-chain warm-up ranking.
- Supervised prediction requires repeated, correctly aligned minute snapshots and both target classes.
- SQLite is appropriate for this single-host deployment, not horizontal multi-writer scaling.

## Validation completed on 2026-09-01

| Check | Result |
|---|---|
| Python compilation for all application packages | Passed |
| Ruff static analysis and formatting | Passed with zero findings |
| SQLite schema initialization | Passed; all nine application tables created |
| FastAPI route/schema validation | Passed; invalid ticker returned `422` |
| Optional API-key enforcement | Passed; missing/wrong keys returned `401`, correct key returned `200` |
| First-snapshot warm-up path | Passed with 10 contracts, one market row, and `cross_sectional_warmup` results |
| Complete temporary pipeline | Passed with 2,660 option rows, 190 stock bars, 2,660 option-feature rows, and 190 market-feature rows |
| Isolation Forest training/persistence | Passed with 2,576 complete scored rows |
| Random Forest training/persistence | Passed on a chronological per-symbol split |
| XGBoost training/persistence/inference | Passed on a chronological per-symbol split |
| API saved-analysis readback | Passed (`200`) |
| Uvicorn process and `/health`/`/docs` | Passed (`200`) |
| Streamlit process and `/_stcore/health` | Passed (`200`) |
| Compose YAML structure | Parsed successfully; `api`, `dashboard`, `scheduler`, and persistent volume present |
| Shipped test/demo function scan | Passed; only the real scheduler process has a `__main__` entry |

The complete pipeline validation used deterministic, market-shaped temporary data solely to exercise code paths; its model metrics are not presented as financial performance. Temporary validation data is not included in the production package.

The live SPY request reached Yahoo, but Yahoo rate-limited the shared validation IP with HTTP `429`. The collector retried three times, FastAPI returned the intended `502` provider error, and the failed run was recorded without fabricating live data. This confirms error handling but prevents claiming a successful live Yahoo snapshot from this environment.

The workspace does not provide a Docker engine, so `docker compose build/up` could not be executed here. The Compose YAML was parsed and the individual Uvicorn and Streamlit processes were started and health-checked directly.
