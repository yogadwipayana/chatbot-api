pm2 stop api
pm2 delete api

git stash
git fetch
git pull

uv venv --python 3.12
uv pip install -e .
.venv/bin/alembic upgrade head

pm2 start .venv/bin/uvicorn --name "api" --interpreter none -- \
  app.main:app --host 0.0.0.0 --port 8000

pm2 save
