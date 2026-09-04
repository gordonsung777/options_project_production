FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG INSTALL_LSTM=false

COPY requirements.txt requirements-lstm.txt ./
RUN pip install --upgrade pip \
    && if [ "$INSTALL_LSTM" = "true" ]; then \
        pip install -r requirements-lstm.txt; \
    else \
        pip install -r requirements.txt; \
    fi

RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --home-dir /app appuser \
    && mkdir -p /app/runtime/models \
    && chown -R appuser:appgroup /app

COPY --chown=appuser:appgroup . .

USER appuser
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
