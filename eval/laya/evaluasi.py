"""Langkah 5: uji checkpoint Laya pada set berlabel (uji.jsonl / kalibrasi.jsonl).

uji.jsonl: 100 pesan = 64 putaran 1 (asal "r1", pembanding JEV) + 36 tambahan
(asal "r2": akronim pendek, riwayat, di luar topik yang mirip akademik).
kalibrasi.jsonl: 60 pesan untuk memilih ambang; jalankan set uji sekali di akhir.
Ringkasan dicetak untuk seluruh set dan per asal.

Tidak meng-import `app`, jadi bisa jalan di Kaggle, di laptop, atau di server.
Dua mode:

    # checkpoint lokal hasil latih_laya.py
    python evaluasi.py --uji data/uji.jsonl --model ./laya-pandu-gerbang
    # multilingual asli (zero-shot), sebagai pembanding
    python evaluasi.py --uji data/uji.jsonl --model basis
    # server Laya yang sedang jalan
    LAYA_API_KEY=... python evaluasi.py --uji data/uji.jsonl \
        --url http://127.0.0.1:8001/v1/systemone

Keputusan blokir dihitung dua kali. Pada Laya, untuk pertanyaan `choice`,
`confidence` adalah entropi ternormalisasi (1 - H/log k) dan TIDAK terkalibrasi;
yang terkalibrasi adalah `answer_confidence` (= probabilitas label terpilih).
`gate.parse_response` membaca `answer_confidence` sejak 2026-10-02 (laya.md A1);
baris `confidence` menunjukkan perilaku gerbang sebelumnya.

Syarat lulus (laya.md): 0 akademik terblokir dan >= 90% di luar topik terblokir,
dibulatkan ke bawah (8/9 pada 64 pesan putaran 1, 18/20 pada set uji penuh).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import time
import urllib.request
from pathlib import Path

KUNCI = "kategori"
LABEL = ["academic", "smalltalk", "out_of_scope", "nonsense", "malicious"]
REPO, SUBFOLDER = "convaiinnovations/laya", "multilingual"
REVISI = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
JEV = (
    "JEV v1 (2026-09-29, 64 pesan x 2): akademik salah label 1/76, akademik terblokir 0, "
    "di luar topik terblokir 18/18, median 0,64 dtk"
)


def blokir(label: str, keyakinan: float, ambang: float, ambang_luar: float) -> bool:
    if label == "academic":
        return False
    return keyakinan >= (ambang_luar if label == "out_of_scope" else ambang)


def ringkas(hasil: list[dict], judul: str) -> None:
    """Akademik terblokir dan di luar topik terblokir, untuk kedua ukuran keyakinan."""
    ak = [h for h in hasil if h["expected"] == "academic"]
    luar = [h for h in hasil if h["expected"] == "out_of_scope"]
    benar = sum(h["choice"] == h["expected"] for h in hasil)
    n = len(hasil)
    print(f"\n{judul}: {n} pesan, akurasi {benar}/{n} ({benar / n:.0%})")
    for nama, kunci in (
        ("confidence (entropi Laya)", "blok_gate"),
        ("answer_confidence (gate.py)", "blok_answer"),
    ):
        a_blok = sum(h[kunci] for h in ak)
        l_blok = sum(h[kunci] for h in luar)
        lulus = a_blok == 0 and l_blok >= math.floor(0.9 * len(luar))
        print(
            f"  {nama:28s}: akademik terblokir {a_blok}/{len(ak)}, di luar topik terblokir "
            f"{l_blok}/{len(luar)} -> {'LULUS' if lulus else 'belum lulus'}"
        )


def prediktor(a):
    if a.url:
        kunci = os.environ.get(a.key_env, "")

        def http(row):
            req = urllib.request.Request(
                a.url, data=json.dumps(row["body"]).encode(), method="POST"
            )
            req.add_header("content-type", "application/json")
            if kunci:
                req.add_header("Authorization", f"Bearer {kunci}")
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["answers"][KUNCI]

        return http

    import laya

    if a.model == "basis":
        agent = laya.load(REPO, subfolder=SUBFOLDER, revision=REVISI, device=a.device)
    else:
        agent = laya.load(a.model, device=a.device)

    def lokal(row):
        return agent.system_one(row["state"], row["questions"])["answers"][KUNCI]

    return lokal


def main() -> None:
    ap = argparse.ArgumentParser(description="Uji Laya pada set uji gerbang PANDU")
    ap.add_argument("--uji", type=Path, required=True)
    ap.add_argument("--model", default=None, help="folder checkpoint, atau 'basis'")
    ap.add_argument("--url", default=None, help="endpoint /v1/systemone (mode HTTP)")
    ap.add_argument("--key-env", default="LAYA_API_KEY")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ambang", type=float, default=0.7)
    ap.add_argument("--ambang-luar", type=float, default=0.9)
    ap.add_argument("--out", type=Path, default=None, help="simpan hasil per pesan (JSONL)")
    a = ap.parse_args()
    if not (a.model or a.url):
        ap.error("isi --model atau --url")

    prediksi = prediktor(a)
    rows = [json.loads(b) for b in a.uji.read_text(encoding="utf-8").splitlines() if b.strip()]
    prediksi(rows[0])  # pemanasan, tidak dihitung latensinya
    hasil = []
    for row in rows:
        t = time.perf_counter()
        j = prediksi(row)
        ms = (time.perf_counter() - t) * 1000
        probs = {k: float(v) for k, v in j["probabilities"].items()}
        conf = float(j.get("confidence", probs[j["choice"]]))
        ans = float(j.get("answer_confidence", probs[j["choice"]]))
        hasil.append(
            {
                "pesan": row["pesan"],
                "unit": row["unit"],
                "asal": row.get("asal", "r1"),
                "riwayat": bool(row.get("riwayat")),
                "expected": row["expected"],
                "choice": j["choice"],
                "confidence": conf,
                "answer_confidence": ans,
                "probabilities": probs,
                "ms": round(ms),
                "blok_gate": blokir(j["choice"], conf, a.ambang, a.ambang_luar),
                "blok_answer": blokir(j["choice"], ans, a.ambang, a.ambang_luar),
            }
        )

    print(f"model: {a.url or a.model}\n")
    print(
        f"{'harapan':14s}"
        + "".join(f"{k[:9]:>10s}" for k in LABEL)
        + "   benar   blok(gate)  blok(answer)"
    )
    for exp in LABEL:
        grup = [h for h in hasil if h["expected"] == exp]
        c = collections.Counter(h["choice"] for h in grup)
        blok_gate = sum(h["blok_gate"] for h in grup)
        blok_answer = sum(h["blok_answer"] for h in grup)
        print(
            f"{exp:14s}"
            + "".join(f"{c[k]:>10d}" for k in LABEL)
            + f"   {c[exp]:>2d}/{len(grup):<3d}  {blok_gate:>3d}/{len(grup):<3d}"
            f"      {blok_answer:>3d}/{len(grup)}"
        )
    ms = sorted(h["ms"] for h in hasil)
    print(f"\nlatensi median {statistics.median(ms):.0f} ms, maks {ms[-1]} ms")
    ringkas(hasil, "semua")
    asal = sorted({h["asal"] for h in hasil})
    if len(asal) > 1:
        for nama in asal:
            ringkas([h for h in hasil if h["asal"] == nama], f"asal {nama}")
    if "r1" in asal:
        print(f"\npembanding untuk asal r1: {JEV}")
    salah = [
        h for h in hasil if h["choice"] != h["expected"] or h["blok_gate"] != h["blok_answer"]
    ]
    if salah:
        print("\nsalah label / beda keputusan:")
        for h in salah:
            print(
                f"  {h['asal'][:3]:3s} {h['expected'][:5]:5s} -> {h['choice'][:9]:9s} "
                f"p={h['answer_confidence']:.2f} H={h['confidence']:.2f} "
                f"{'BLOK' if h['blok_answer'] else '    '} {'R' if h['riwayat'] else ' '} "
                f"{h['pesan'][:60]}"
            )
    if a.out:
        a.out.write_text(
            "".join(json.dumps(h, ensure_ascii=False) + "\n" for h in hasil), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
