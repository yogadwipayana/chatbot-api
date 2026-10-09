uv venv --python 3.12
uv pip install -e .
# Bukan `alembic upgrade head` langsung: database yang lebih baru dari kode ini
# dilewati dengan peringatan, bukan gagal start (app/db/migrasi.py).
.venv/bin/python -m app.db.migrasi

pm2 start .venv/bin/uvicorn --name "api" --interpreter none -- \
  app.main:app --host 0.0.0.0 --port 8000

pm2 save
