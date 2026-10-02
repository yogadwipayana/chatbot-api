"""Langkah 2: beri label JEV untuk setiap pesan latih (JEV sebagai guru, laya.md).

Body permintaan dibuat dengan `gate.build_request`, jadi sama persis dengan yang
dikirim gerbang di produksi (instruksi, kriteria, `topik_dipilih`, `riwayat`).
Jawaban JEV (probabilitas per label) menjadi target lunak (`gold`) untuk Laya.
JEV dipanggil lewat gateway (`JEV_URL`, bawaan `<BASE_URL>/systemone`), tidak
langsung ke OpenRouter.

Bisa dilanjutkan bila terputus: id yang sudah berlabel dilewati. Pesan yang
gagal setelah beberapa kali coba ditulis dengan `ok: false` dan dicoba lagi
pada jalan berikutnya.

    python -m eval.laya.label_jev --pesan eval/laya/data/pesan.jsonl \
        --out eval/laya/data/label_jev.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

from app.config import get_settings
from app.rag import gate as G
from app.rag.providers import USER_AGENT

BIAYA_PER_PANGGILAN = 0.000023
"""Perkiraan biaya JEV lewat gateway (ai-gateway-facts, 2026-09-25)."""


def baca_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(b) for b in path.read_text(encoding="utf-8").splitlines() if b.strip()]


async def label_satu(
    client: httpx.AsyncClient, url: str, headers: dict, model: str, row: dict, coba: int
) -> dict:
    body = G.build_request(
        model, row["pesan"], [tuple(t) for t in row["riwayat"]], row["unit"]
    )
    galat = ""
    for i in range(coba):
        t = time.perf_counter()
        try:
            r = await client.post(url, json=body, headers=headers)
            r.raise_for_status()
            j = r.json()["answers"][G.QUESTION_KEY]
            return {
                "id": row["id"],
                "ok": True,
                "choice": j["choice"],
                "confidence": j.get("confidence"),
                "probabilities": {
                    k: float(v) for k, v in (j.get("probabilities") or {}).items()
                },
                "ms": round((time.perf_counter() - t) * 1000),
            }
        except Exception as exc:  # noqa: BLE001 -- dicoba ulang, lalu dicatat
            galat = f"{type(exc).__name__}: {exc}"[:200]
            await asyncio.sleep(2 * (i + 1))
    return {"id": row["id"], "ok": False, "error": galat}


async def utama(pesan: Path, out: Path, paralel: int, timeout: float, coba: int) -> None:
    s = get_settings()
    url, model = s.url_jev(), s.jev_model
    kunci = s.kunci_jev()
    if not url or kunci is None:
        raise SystemExit("JEV_URL/BASE_URL dan JEV_API_KEY/API_KEY harus terisi di api/.env")
    headers = {"Authorization": f"Bearer {kunci.get_secret_value()}", "User-Agent": USER_AGENT}

    rows = baca_jsonl(pesan)
    sudah = {r["id"] for r in baca_jsonl(out) if r.get("ok")}
    sisa = [r for r in {r["id"]: r for r in rows}.values() if r["id"] not in sudah]
    print(
        f"{len(rows)} pesan, {len(sudah)} sudah berlabel, {len(sisa)} dilabeli "
        f"(perkiraan biaya ${len(sisa) * BIAYA_PER_PANGGILAN:.3f}); model {model}"
    )

    sem = asyncio.Semaphore(paralel)
    kunci_tulis = asyncio.Lock()
    hitung = {"ok": 0, "gagal": 0}
    mulai = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:

        async def satu(row: dict) -> None:
            async with sem:
                hasil = await label_satu(client, url, headers, model, row, coba)
            async with kunci_tulis:
                with out.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(hasil, ensure_ascii=False) + "\n")
                hitung["ok" if hasil["ok"] else "gagal"] += 1
                n = hitung["ok"] + hitung["gagal"]
                if n % 50 == 0 or n == len(sisa):
                    laju = n / max(1e-9, time.perf_counter() - mulai)
                    print(
                        f"  {n}/{len(sisa)}  ok={hitung['ok']} gagal={hitung['gagal']}  "
                        f"{laju:.1f}/dtk",
                        flush=True,
                    )

        await asyncio.gather(*(satu(r) for r in sisa))
    print(f"SELESAI: ok={hitung['ok']} gagal={hitung['gagal']} -> {out}")


def main() -> None:
    logging.disable(logging.WARNING)
    p = argparse.ArgumentParser(description="Label JEV untuk pesan latih Laya")
    p.add_argument("--pesan", type=Path, default=Path("eval/laya/data/pesan.jsonl"))
    p.add_argument("--out", type=Path, default=Path("eval/laya/data/label_jev.jsonl"))
    p.add_argument(
        "--paralel",
        type=int,
        default=2,
        help="JEV lewat gateway sering timeout bila >3 panggilan bersamaan (2026-09-28)",
    )
    p.add_argument("--timeout", type=float, default=45.0)
    p.add_argument("--coba", type=int, default=3)
    a = p.parse_args()
    asyncio.run(utama(a.pesan, a.out, a.paralel, a.timeout, a.coba))


if __name__ == "__main__":
    main()
