ssh -N -L 5432:127.0.0.1:5432 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes root@ai.dwipa.my.id

ssh root@ai.dwipa.my.id cat /root/pgweb-credentials.txt

uv venv --python 3.12
uv pip install -e ".[dev]"

.venv\Scripts\activate
uvicorn app.main:app --reload
