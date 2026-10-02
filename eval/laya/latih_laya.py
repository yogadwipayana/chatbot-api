"""Langkah 4: fine-tune Laya multilingual untuk gerbang PANDU (laya.md).

Dijalankan di mesin ber-GPU (Kaggle/Colab), BUKAN di server (tanpa GPU, CPU tanpa
AVX). Tidak meng-import `app`: hanya butuh `laya`, torch, dan dataset hasil
`bangun_dataset.py` (train.jsonl, val.jsonl, manifest.json).

Diadaptasi dari skrip resmi `notebooks/laya_finetune_typed_decisions_mps.py`
(Laya v0.3.22): loss RLCD yang sama (policy gradient ala GRPO dengan reward
proper scoring rule + cross-entropy terhadap target lunak), LR encoder 2,5e-5 dan
head 1e-4, sigma 0,4 -> 0,1. Tambahan di sini: CUDA + AMP fp16 (T4 tidak punya
bf16), metrik validasi per epoch, bobot terbaik menurut CE validasi, dan
kalibrasi temperature pada set validasi (bukan potongan data latih).

Titik awal dipatok ke checkpoint yang sama dengan server:
convaiinnovations/laya, subfolder multilingual, revisi 55cf4c4e.

    pip install "laya @ git+https://github.com/NandhaKishorM/laya@v0.3.22"
    python latih_laya.py --data ./data --out ./laya-pandu-gerbang
    python latih_laya.py --data ./data --out ./uji-asap --limit 32 --epochs 1 \
        --device cpu --freeze-encoder

Keluaran di --out: model.safetensors (fp16), encoder/, tokenizer/,
rl_agent_config.json (temperature terkalibrasi), laporan_latih.json.
Muat dengan `laya.load(<out>)`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, TEMP_MAX, TEMP_MIN, build_model, build_sequence, proper_reward
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

REPO = "convaiinnovations/laya"
SUBFOLDER = "multilingual"
REVISI = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
KUNCI = "kategori"
AMBANG_BLOKIR, AMBANG_LUAR = 0.7, 0.9
"""GatePolicy produksi, hanya untuk metrik validasi."""


def baca_jsonl(path: Path) -> list[dict]:
    return [json.loads(b) for b in path.read_text(encoding="utf-8").splitlines() if b.strip()]


def siapkan_model(cache: Path | None) -> Path:
    """Unduh checkpoint basis; tanpa --cache memakai cache HF (HF_HOME), yang di server
    sudah berisi revisi yang sama (volume laya-data), jadi tidak diunduh ulang."""
    kw = {"local_dir": str(cache)} if cache else {}
    root = snapshot_download(REPO, revision=REVISI, allow_patterns=[f"{SUBFOLDER}/*"], **kw)
    d = Path(root) / SUBFOLDER
    _fix_tokenizer_config(str(d))
    return d


def buat_item(tok, cfg: dict, row: dict) -> dict | None:
    q = row["questions"][KUNCI]
    crit = q["criteria"]
    gold = row["gold"][KUNCI]["probabilities"]
    target = [float(gold.get(k, 0.0)) for k in crit]
    total = sum(target)
    target = [x / total for x in target] if total > 0 else [1.0 / len(target)] * len(target)
    ids, markers = build_sequence(
        tok,
        row["state"],
        {"t": "choice", "ins": q["instructions"], "crit": crit},
        cfg["max_len"],
        cfg["head_max_len"],
    )
    if len(markers) != len(crit):
        return None
    return {
        "ids": ids,
        "markers": markers,
        "qtype": QTYPES["choice"],
        "target": target,
        "label": target.index(max(target)),
        "keys": list(crit),
    }


def collate(items: list[dict], pad_id: int):
    n, seq = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, seq), pad_id, dtype=torch.long)
    att = torch.zeros((n, seq), dtype=torch.long)
    pos = torch.zeros((n, kmax), dtype=torch.long)
    mask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        pos[i, :k] = torch.tensor(it["markers"])
        mask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"])
    return ids, att, pos, mask, target, torch.tensor([it["qtype"] for it in items])


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Sama dengan `fit_temperature` resmi: minimalkan CE terhadap target lunak."""
    if len(logits) < 10:
        return 1.0
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(targets * torch.log_softmax(logits / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    # Rentang yang dipakai Agent saat memuat (clamp_temperature); di luar itu dipangkas.
    return float(torch.clamp(log_t.exp(), TEMP_MIN, TEMP_MAX).item())


@torch.no_grad()
def logit_semua(model, items, pad_id, device, batch, amp) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    semua_z, semua_t = [], []
    for s in range(0, len(items), batch):
        ids, att, pos, mask, target, qtype = collate(items[s : s + batch], pad_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            z, _ = model(
                ids.to(device),
                att.to(device),
                pos.to(device),
                mask.to(device),
                qtype.to(device),
            )
        z = z.float().masked_fill(~mask.to(device), -1e4).cpu()
        semua_z.append(z)
        semua_t.append(target)
    model.train()
    return torch.cat(semua_z), torch.cat(semua_t)


def metrik(z: torch.Tensor, t: torch.Tensor, keys: list[str], suhu: float = 1.0) -> dict:
    p = torch.softmax(z / suhu, -1)
    ce = float(-(t * torch.log_softmax(z / suhu, -1)).sum(-1).mean())
    pred, emas = p.argmax(-1), t.argmax(-1)
    akad = keys.index("academic")
    luar = keys.index("out_of_scope")
    pmax = p.max(-1).values
    ambang = torch.where(pred == luar, torch.tensor(AMBANG_LUAR), torch.tensor(AMBANG_BLOKIR))
    blok = (pred != akad) & (pmax >= ambang)
    per_label = {
        k: f"{int(((pred == i) & (emas == i)).sum())}/{int((emas == i).sum())}"
        for i, k in enumerate(keys)
    }
    return {
        "ce": round(ce, 4),
        "akurasi": round(float((pred == emas).float().mean()), 4),
        "akademik_terblokir": int((blok & (emas == akad)).sum()),
        "akademik_total": int((emas == akad).sum()),
        "benar_per_label": per_label,
    }


def simpan(model, tok, cfg: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items()},
        str(out / "model.safetensors"),
    )
    model.encoder.config.save_pretrained(out / "encoder")
    tok.save_pretrained(out / "tokenizer")
    (out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune Laya multilingual untuk gerbang PANDU")
    ap.add_argument(
        "--data", type=Path, required=True, help="folder train.jsonl, val.jsonl, manifest.json"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None, help="folder unduhan; bawaan cache HF")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--no-amp", action="store_true", help="matikan fp16 autocast di CUDA")
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument(
        "--freeze-encoder", action="store_true", help="uji asap di CPU: latih head saja"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="uji asap: pakai N baris latih/validasi"
    )
    ap.add_argument("--seed", type=int, default=20260930)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(a.device)
    amp = device.type == "cuda" and not a.no_amp

    model_dir = siapkan_model(a.cache)
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")

    train_rows = baca_jsonl(a.data / "train.jsonl")
    val_rows = baca_jsonl(a.data / "val.jsonl")
    if a.limit:
        train_rows, val_rows = train_rows[: a.limit], val_rows[: max(10, a.limit // 4)]
    train = [it for it in (buat_item(tok, cfg, r) for r in train_rows) if it]
    val = [it for it in (buat_item(tok, cfg, r) for r in val_rows) if it]
    keys = train[0]["keys"]
    panjang = sorted(len(it["ids"]) for it in train)
    print(
        f"device={device} amp={amp}; train {len(train)}/{len(train_rows)}, "
        f"val {len(val)}/{len(val_rows)}; "
        f"token median {panjang[len(panjang) // 2]}, maks {panjang[-1]}; label {keys}"
    )

    model = build_model(cfg, encoder_dir=model_dir / "encoder")
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model.float()
    if not a.no_checkpointing and not a.freeze_encoder:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.head_checkpointing = True
    if a.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    model.to(device).train()

    enc = [p for n, p in model.named_parameters() if "encoder." in n and p.requires_grad]
    head = [p for n, p in model.named_parameters() if "encoder." not in n]
    groups = [{"params": head, "lr": a.lr_head}] + (
        [{"params": enc, "lr": a.lr_encoder}] if enc else []
    )
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    updates = max(1, math.ceil(len(train) / a.micro_batch / a.grad_accum) * a.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    z0, t0 = logit_semua(model, val, tok.pad_token_id, device, a.micro_batch, amp)
    laporan = {
        "basis": f"{REPO}/{SUBFOLDER}@{REVISI}",
        "argumen": {k: str(v) for k, v in vars(a).items()},
        "epoch": [{"epoch": 0, "val": metrik(z0, t0, keys)}],
    }
    print("epoch 0 (tanpa latih) val:", laporan["epoch"][0]["val"])

    terbaik, bobot_terbaik = float("inf"), None
    mulai = time.time()
    for ep in range(a.epochs):
        random.Random(a.seed + ep).shuffle(train)
        sigma = 0.4 + (0.1 - 0.4) * ep / max(1, a.epochs - 1)
        opt.zero_grad(set_to_none=True)
        total, nb = 0.0, 0
        for s in range(0, len(train), a.micro_batch):
            ids, att, pos, mask, target, qtype = (
                x.to(device) for x in collate(train[s : s + a.micro_batch], tok.pad_token_id)
            )
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits, act = model(ids, att, pos, mask, qtype)
            logits = logits.float()
            k = mask.sum(-1, keepdim=True).float()
            eps = torch.randn((4,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            noisy = logits.detach().unsqueeze(0) + eps
            probs = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                reward = proper_reward(
                    probs, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0
                )
                adv = reward - reward.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = (
                -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1))
                .sum(-1)
                .mean()
            )
            loss = (loss_rl + loss_ce + 0.0 * act.float().sum()) / a.grad_accum
            scaler.scale(loss).backward()
            nb += 1
            if nb % a.grad_accum == 0 or s + a.micro_batch >= len(train):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            total += loss.item() * a.grad_accum
            if nb % 50 == 0:
                print(
                    f"  epoch {ep + 1} step {nb} loss {loss.item() * a.grad_accum:.4f} "
                    f"({time.time() - mulai:.0f} dtk)",
                    flush=True,
                )
        z, t = logit_semua(model, val, tok.pad_token_id, device, a.micro_batch, amp)
        m = metrik(z, t, keys)
        laporan["epoch"].append(
            {"epoch": ep + 1, "loss_latih": round(total / max(1, nb), 4), "val": m}
        )
        print(f"epoch {ep + 1}/{a.epochs} loss {total / max(1, nb):.4f} val {m}", flush=True)
        if m["ce"] < terbaik:
            terbaik = m["ce"]
            # fp16 di CPU: checkpoint akhir memang disimpan fp16, jadi tidak ada presisi
            # yang hilang, RAM separuh, dan kalibrasi di bawah memakai bobot yang sama
            # persis dengan yang dimuat saat inferensi.
            bobot_terbaik = {
                k: v.detach().to("cpu", torch.float16, copy=True)
                if v.is_floating_point()
                else v.detach().to("cpu", copy=True)
                for k, v in model.state_dict().items()
            }
            laporan["epoch_terbaik"] = ep + 1

    if bobot_terbaik is not None:
        model.load_state_dict(bobot_terbaik)
        del bobot_terbaik
    z, t = logit_semua(model, val, tok.pad_token_id, device, a.micro_batch, amp)
    suhu = fit_temperature(z, t)
    laporan["temperature_choice"] = suhu
    laporan["val_terkalibrasi"] = metrik(z, t, keys, suhu)
    laporan["menit"] = round((time.time() - mulai) / 60, 1)
    print(f"temperature choice {suhu:.3f}; val terkalibrasi {laporan['val_terkalibrasi']}")

    manifest_path = a.data / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    temps[QTYPES["choice"]] = suhu
    cfg.update({"fine_tuned": True, "model_name": "laya-pandu-gerbang", "temperature": temps})
    cfg.pop("temperature_by_options", None)
    cfg["training"] = {
        "basis": laporan["basis"],
        "epochs": a.epochs,
        "epoch_terbaik": laporan.get("epoch_terbaik"),
        "n_train": len(train),
        "n_val": len(val),
        "fine_tuned_from_checkpoint": True,
    }
    cfg["pandu"] = {
        "gerbang_sha256": manifest.get("gerbang_sha256"),
        "guru": manifest.get("guru"),
        "data_sha256": manifest.get("sha256"),
        "train_sha256": hashlib.sha256((a.data / "train.jsonl").read_bytes()).hexdigest(),
    }
    simpan(model, tok, cfg, a.out)
    (a.out / "laporan_latih.json").write_text(
        json.dumps(laporan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"SELESAI -> {a.out} ({laporan['menit']} menit)")


if __name__ == "__main__":
    main()
