FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# uv + uv.lock agar versi dependensi identik dengan pengembangan lokal. Versi uv
# dipatok (versi yang menulis uv.lock): `latest` membuat build hari ini dan besok
# bisa berbeda tanpa satu baris pun berubah di repo.
COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /usr/local/bin/uv

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
# Pengguna `app` (uid/gid 10001) menjalankan migrasi dan uvicorn. Hanya `log/`
# dan `storage/` yang boleh ditulisnya; kode dan venv tetap milik root.
# `setpriv` (util-linux) dipakai docker-entrypoint.sh untuk melepas hak root:
# build gagal di sini bila ia tidak ada, bukan kontainer gagal saat start.
RUN EXTRAS=$([ "$INSTALL_LOCAL_MODELS" = "true" ] && echo "--extra local"); \
    uv sync --frozen --no-dev $EXTRAS \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app \
       --shell /usr/sbin/nologin app \
    && mkdir -p log storage/documents \
    && chown -R app:app log storage \
    && command -v setpriv

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

# Entrypoint: rapikan pemilik volume sebagai root, lepas hak root, migrasi
# (RUN_MIGRATIONS=false untuk melewatinya), lalu jalankan CMD sebagai `app`.
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
