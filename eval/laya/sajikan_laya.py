"""Langkah 6: sajikan checkpoint hasil fine-tuning lewat `laya-serve` (laya.md).

`laya-serve` hanya mengenal checkpoint bawaan. Skrip ini menjalankan server
yang sama persis (`laya.serve.main`: auth, batas request, /health,
/v1/systemone), hanya router-nya diganti supaya nama `multilingual` menunjuk ke
folder checkpoint lokal. Gerbang chatbot tetap mengirim `JEV_MODEL=multilingual`.

SIGILL di CPU server (QEMU tanpa AVX): sekitar 13% proses torch mati pada
prediksi pertamanya karena MKL memilih kernel AVX-512
`mkl_vml_kernel_sCos_Z0HAynn` (dipanggil `torch.cos` di rotary embedding).
Pilihan itu dibuat sekali per proses. Uji 2026-09-30, 30 proses baru per varian:
tanpa perbaikan 4 crash, MKL_ENABLE_INSTRUCTIONS=SSE4_2 5, MKL_CBWR=SSE4_2 4,
MKL_CBWR=COMPATIBLE 0 (latensi sama: median 996 vs 999 ms). Dua lapis di sini:
1. MKL_CBWR=COMPATIBLE dipasang sebelum torch dimuat (bisa ditimpa lewat env);
2. satu prediksi pemanasan SEBELUM server menerima request: bila tetap crash,
   `restart: always` menyalakan ulang container, dan /health baru menjawab
   setelah pemanasan lolos, jadi tidak ada request mahasiswa yang kena.

Env: LAYA_PANDU_PATH (folder hasil latih_laya.py; kosong = checkpoint
multilingual asli), selebihnya sama dengan laya-serve (LAYA_API_KEY,
LAYA_THREADS, LAYA_DEVICE, LAYA_PORT, ...).

    LAYA_PANDU_PATH=/models/laya-pandu-gerbang python sajikan_laya.py
"""

from __future__ import annotations

import logging
import os
import time

os.environ.setdefault("MKL_CBWR", "COMPATIBLE")  # harus sebelum torch dimuat

from laya import serve  # noqa: E402
from laya.router import Router  # noqa: E402

log = logging.getLogger("sajikan_laya")

PEMANASAN_STATE = {"pesan_terbaru": "pemanasan", "topik_dipilih": "BAAK"}
PEMANASAN_TANYA = {
    "kategori": {
        "type": "choice",
        "instructions": "Pemanasan server.",
        "criteria": {"a": "pilihan pertama", "b": "pilihan kedua"},
    }
}


def build_router() -> Router:
    from laya.mcp.device import env_device

    serve._apply_thread_limit()
    path = os.environ.get("LAYA_PANDU_PATH", "").strip()
    models = {}
    if path:
        if not os.path.isfile(os.path.join(path, "model.safetensors")):
            raise SystemExit(f"LAYA_PANDU_PATH={path!r} bukan folder checkpoint Laya")
        models["multilingual"] = path
    router = Router(device=env_device(), models=models, max_loaded=1)
    router.preload(["multilingual"])
    t = time.perf_counter()
    router.predict(PEMANASAN_STATE, PEMANASAN_TANYA, model="multilingual")
    print(
        f"sajikan_laya: pemanasan lolos ({time.perf_counter() - t:.1f} dtk), "
        f"checkpoint {path or 'multilingual asli'}, MKL_CBWR={os.environ.get('MKL_CBWR')}",
        flush=True,
    )
    return router


if __name__ == "__main__":
    serve.build_router = build_router
    serve.main()
