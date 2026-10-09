#!/bin/sh
# Titik masuk kontainer api.
#
# Mulai sebagai root HANYA untuk merapikan pemilik `log/` dan `storage/`: volume
# bind-mount dari host bisa dimiliki root, dan proses non-root tidak dapat
# menulis log SQLite maupun PDF ke sana. Sesudah itu hak root dilepas
# (`setpriv`), dan migrasi serta uvicorn berjalan sebagai pengguna `app`.
set -eu

if [ "$(id -u)" = "0" ]; then
    for dir in log storage; do
        mkdir -p "$dir"
        # Hanya yang belum milik `app`: restart berikutnya tidak menyentuh apa pun.
        if ! find "$dir" ! -user app -exec chown app:app {} + ; then
            echo "peringatan: pemilik $dir tidak dapat diubah; log atau unggahan bisa gagal ditulis" >&2
        fi
    done
    export HOME=/home/app
    exec setpriv --reuid=10001 --regid=10001 --clear-groups "$0" "$@"
fi

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
    # Bukan `alembic upgrade head` langsung: database yang lebih baru dari image
    # ini dilewati dengan peringatan, bukan restart-loop (app/db/migrasi.py).
    python -m app.db.migrasi
fi

exec "$@"
