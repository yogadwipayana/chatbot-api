"""Langkah 3: gabungkan pesan + label JEV menjadi dataset latih Laya (laya.md).

- Label: distribusi probabilitas JEV (target lunak). Pesan yang label JEV-nya
  (argmax) tidak sama dengan `niat` dari `buat_pesan.py` masuk `tinjau.jsonl`
  dan secara bawaan TIDAK ikut dilatih (`--tidak-setuju`). File itu juga cara
  menemukan kesalahan JEV (misalnya sertifikasi Adobe dilabeli out_of_scope
  0,90). Salin ke `keputusan.jsonl`, tambahkan `"keputusan": "<label>"` atau
  `"keputusan": "buang"` pada baris yang sudah ditinjau, lalu bangun ulang:
  baris itu dilatih dengan label hasil tinjauan (target one-hot yang dihaluskan).
- Pesan yang sama (setelah dinormalisasi) hanya dipakai sekali.
- Pesan yang sama atau mirip dengan set uji (`set_uji.py`) dibuang agar angka
  uji tidak bocor.
- `state` dan `questions` dibuat dengan `gate.build_request`, jadi sama persis
  dengan yang dikirim gerbang di produksi. Laya membaca teks instruksi dan
  kriteria sebagai masukan: bila CRITERIA/INSTRUCTIONS di gate.py berubah,
  bangun ulang dataset dan latih ulang (lihat `gerbang_sha256` di manifest).

Masukan di `--dir`: pesan.jsonl, label_jev.jsonl, keputusan.jsonl (opsional).
Keluaran di `--dir`: train.jsonl, val.jsonl, uji.jsonl, tinjau.jsonl, manifest.json.
Satu baris train/val:

    {"id": ..., "state": {...}, "questions": {"kategori": {...}},
     "gold": {"kategori": {"probabilities": {"academic": 0.97, ...}}},
     "label": "academic", "niat": "academic", "sumber": "topik"}

    python -m eval.laya.bangun_dataset --dir eval/laya/data
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.rag import gate as G
from eval.laya.set_uji import SET

LABEL = [str(k) for k in G.CRITERIA]
MODEL_LAYA = "multilingual"
"""Isi field `model` di body. Tidak dibaca saat latih; dipakai saat evaluasi via HTTP."""


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.casefold()).split())


def jaccard(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    return len(x & y) / len(x | y) if x and y else 0.0


def baca_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(b) for b in path.read_text(encoding="utf-8").splitlines() if b.strip()]


def tulis_jsonl(path: Path, rows: list[dict]) -> str:
    teks = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(teks, encoding="utf-8", newline="\n")
    return hashlib.sha256(teks.encode()).hexdigest()


def baris_laya(row: dict, probs: dict[str, float]) -> dict:
    body = G.build_request(
        MODEL_LAYA, row["pesan"], [tuple(t) for t in row["riwayat"]], row["unit"]
    )
    total = sum(probs.get(k, 0.0) for k in LABEL) or 1.0
    gold = {k: round(probs.get(k, 0.0) / total, 6) for k in LABEL}
    return {
        "id": row["id"],
        "state": body["state"],
        "questions": body["questions"],
        "gold": {G.QUESTION_KEY: {"probabilities": gold}},
        "label": max(gold, key=gold.get),
        "niat": row["niat"],
        "sumber": row["sumber"],
    }


def target_niat(niat: str, halus: float = 0.1) -> dict[str, float]:
    sisa = halus / (len(LABEL) - 1)
    return {k: (1.0 - halus if k == niat else sisa) for k in LABEL}


def utama(
    d: Path,
    pesan: Path,
    label: Path,
    keputusan_path: Path,
    tidak_setuju: str,
    val_frac: float,
    seed: int,
) -> None:
    rows = {r["id"]: r for r in baca_jsonl(pesan)}
    keputusan = {
        r["id"]: r["keputusan"] for r in baca_jsonl(keputusan_path) if r.get("keputusan")
    }
    salah = {v for v in keputusan.values() if v not in (*LABEL, "buang")}
    if salah:
        raise SystemExit(f"keputusan tidak dikenal: {sorted(salah)}; pakai {LABEL} atau buang")
    labels: dict[str, dict] = {}
    for r in baca_jsonl(label):
        if r.get("ok"):
            labels[r["id"]] = r  # yang terakhir menang
    uji_norm = [norm(p) for _, _, p in SET]
    uji_set = set(uji_norm)

    stat = collections.Counter()
    setuju_per_niat: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    terpakai, tinjau = [], []
    dilihat: set[tuple] = set()
    for rid, row in rows.items():
        lab = labels.get(rid)
        if lab is None:
            stat["tanpa_label"] += 1
            continue
        n = norm(row["pesan"])
        if not n and row["niat"] != "nonsense":
            stat["kosong"] += 1
            continue
        if n in uji_set or (
            len(n.split()) >= 2 and any(jaccard(n, u) >= 0.8 for u in uji_norm)
        ):
            stat["mirip_uji"] += 1
            continue
        kunci = (
            n or row["pesan"],
            row["unit"],
            json.dumps(row["riwayat"], ensure_ascii=False),
        )
        if kunci in dilihat:
            stat["duplikat"] += 1
            continue
        probs = lab["probabilities"]
        jev = max(probs, key=probs.get) if probs else lab["choice"]
        setuju = jev == row["niat"]
        setuju_per_niat[row["niat"]][jev] += 1
        if setuju:
            dilihat.add(kunci)
            terpakai.append(baris_laya(row, probs))
            continue
        tinjau.append(
            {
                "id": rid,
                "pesan": row["pesan"],
                "unit": row["unit"],
                "riwayat": row["riwayat"],
                "niat": row["niat"],
                "jev": jev,
                "jev_probabilities": probs,
                "sumber": row["sumber"],
                **({"keputusan": keputusan[rid]} if rid in keputusan else {}),
            }
        )
        if rid in keputusan:
            stat["ditinjau"] += 1
            if keputusan[rid] != "buang":
                dilihat.add(kunci)
                terpakai.append(baris_laya(row, target_niat(keputusan[rid])))
        elif tidak_setuju == "jev":
            dilihat.add(kunci)
            terpakai.append(baris_laya(row, probs))
        elif tidak_setuju == "niat":
            dilihat.add(kunci)
            terpakai.append(baris_laya(row, target_niat(row["niat"])))

    rng = random.Random(seed)
    per_label: dict[str, list[dict]] = collections.defaultdict(list)
    for r in terpakai:
        per_label[r["label"]].append(r)
    train, val = [], []
    for lab in LABEL:
        grup = per_label.get(lab, [])
        rng.shuffle(grup)
        k = max(1, round(len(grup) * val_frac)) if len(grup) >= 5 else 0
        val += grup[:k]
        train += grup[k:]
    rng.shuffle(train)
    rng.shuffle(val)

    uji = []
    for i, (harap, unit, p) in enumerate(SET):
        body = G.build_request(MODEL_LAYA, p, (), unit)
        uji.append(
            {
                "i": i,
                "pesan": p,
                "unit": unit,
                "expected": harap,
                "state": body["state"],
                "questions": body["questions"],
                "body": body,
            }
        )

    d.mkdir(parents=True, exist_ok=True)
    pertanyaan = G.build_request(MODEL_LAYA, "x", (), None)["questions"]
    manifest = {
        "dibuat": datetime.now(UTC).isoformat(timespec="seconds"),
        "guru": get_settings().jev_model,
        "label": LABEL,
        "gerbang_sha256": hashlib.sha256(
            json.dumps(pertanyaan, sort_keys=True).encode()
        ).hexdigest(),
        "tidak_setuju": tidak_setuju,
        "jumlah": {
            "pesan": len(rows),
            "train": len(train),
            "val": len(val),
            "uji": len(uji),
            "tinjau": len(tinjau),
            **dict(stat),
        },
        "train_per_label": dict(collections.Counter(r["label"] for r in train)),
        "val_per_label": dict(collections.Counter(r["label"] for r in val)),
        "niat_vs_jev": {k: dict(v) for k, v in setuju_per_niat.items()},
        "sha256": {
            "train.jsonl": tulis_jsonl(d / "train.jsonl", train),
            "val.jsonl": tulis_jsonl(d / "val.jsonl", val),
            "uji.jsonl": tulis_jsonl(d / "uji.jsonl", uji),
        },
    }
    tulis_jsonl(d / "tinjau.jsonl", tinjau)
    (d / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    # Satu file untuk diunggah sebagai Kaggle Dataset (latih_kaggle.ipynb).
    skrip = Path(__file__).parent
    with zipfile.ZipFile(d / "kaggle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for nama in ("latih_laya.py", "evaluasi.py"):
            z.write(skrip / nama, nama)
        for nama in ("train.jsonl", "val.jsonl", "uji.jsonl", "manifest.json"):
            z.write(d / nama, nama)

    print(
        json.dumps(
            {k: manifest[k] for k in ("jumlah", "train_per_label", "val_per_label")},
            ensure_ascii=False,
            indent=1,
        )
    )
    print("niat -> label JEV (baris = niat):")
    for niat in LABEL:
        c = setuju_per_niat.get(niat, collections.Counter())
        tot = sum(c.values())
        if tot:
            print(
                f"  {niat:13s} setuju {c[niat]:4d}/{tot:<4d} ({c[niat] / tot:.0%})  "
                + ", ".join(f"{k}={v}" for k, v in c.most_common() if k != niat)
            )
    print(f"-> {d} (unggah {d / 'kaggle.zip'} sebagai Kaggle Dataset)")


def main() -> None:
    p = argparse.ArgumentParser(description="Bangun dataset latih Laya dari pesan + label JEV")
    p.add_argument("--dir", type=Path, default=Path("eval/laya/data"))
    p.add_argument("--pesan", type=Path, default=None, help="bawaan: <dir>/pesan.jsonl")
    p.add_argument("--label", type=Path, default=None, help="bawaan: <dir>/label_jev.jsonl")
    p.add_argument(
        "--keputusan", type=Path, default=None, help="bawaan: <dir>/keputusan.jsonl"
    )
    p.add_argument(
        "--tidak-setuju",
        choices=["buang", "jev", "niat"],
        default="buang",
        help="label JEV beda dari niat: buang, pakai label JEV, atau pakai niat",
    )
    p.add_argument("--val", type=float, default=0.1, help="porsi validasi per label")
    p.add_argument("--seed", type=int, default=20260930)
    a = p.parse_args()
    utama(
        a.dir,
        a.pesan or a.dir / "pesan.jsonl",
        a.label or a.dir / "label_jev.jsonl",
        a.keputusan or a.dir / "keputusan.jsonl",
        a.tidak_setuju,
        a.val,
        a.seed,
    )


if __name__ == "__main__":
    main()
