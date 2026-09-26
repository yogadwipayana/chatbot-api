FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# uv + uv.lock agar versi dependensi identik dengan pengembangan lokal.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# true = ikut pasang extra `local` (sentence-transformers + torch, ratusan MB),
# hanya perlu bila EMBED_PROVIDER=local atau RERANK_PROVIDER=local.
ARG INSTALL_LOCAL_MODELS=false

# Dependensi dulu (layer ter-cache selama pyproject/uv.lock tidak berubah).
COPY pyproject.toml uv.lock ./
RUN EXTRAS=$([ "$INSTALL_LOCAL_MODELS" = "true" ] && echo "--extra local"); \
    uv sync --frozen --no-dev --no-install-project $EXTRAS

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY eval ./eval
COPY scripts ./scripts
RUN EXTRAS=$([ "$INSTALL_LOCAL_MODELS" = "true" ] && echo "--extra local"); \
    uv sync --frozen --no-dev $EXTRAS \
    && mkdir -p log storage/documents

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

# Migrasi dijalankan saat start seperti start.sh; RUN_MIGRATIONS=false untuk melewatinya.
CMD ["sh", "-c", "if [ \"${RUN_MIGRATIONS:-true}\" = \"true\" ]; then alembic upgrade head; fi && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
