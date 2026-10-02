# Fine-tuning Laya untuk gerbang PANDU

Pipeline untuk melatih `laya-multilingual` (mmBERT-base, 322M) menjadi pengganti lokal
JEV di gerbang (`app/rag/gate.py`). Latar belakang, hasil uji, dan keputusan ada di
`laya.md` (root proyek). Tanpa fine-tuning Laya hanya benar 9/64 pada set uji gerbang;
README Laya sendiri menyebut checkpoint dasarnya "near chance zero-shot".

```
buat_pesan.py ──> label_jev.py ──> bangun_dataset.py ──> latih_kaggle.ipynb ──> evaluasi.py ──> sajikan_laya.py
 (LLM gateway)    (JEV = guru)     (+ keputusan.jsonl)   (GPU: latih_laya.py)   (64 pesan uji)   (server /laya)
   lokal             lokal              lokal                 Kaggle/Colab        di mana saja      server
```

| File | Isi |
|---|---|
| `set_uji.py` | 64 pesan berlabel tangan (jev.md §8). Hanya untuk uji, tidak pernah dilatih |
| `buat_pesan.py` | Langkah 1: pesan sintetis lewat LLM chat gateway (`build_llm`) |
| `label_jev.py` | Langkah 2: label JEV lewat gateway, body dari `gate.build_request` |
| `bangun_dataset.py` | Langkah 3: saring, dedup, buang yang mirip set uji, split, tulis dataset |
| `latih_laya.py` | Langkah 4: training (RLCD resmi Laya + AMP, validasi, kalibrasi). Tanpa `app` |
| `latih_kaggle.ipynb` | Notebook Kaggle yang menjalankan langkah 4 dan 5 |
| `evaluasi.py` | Langkah 5: uji checkpoint lokal / server HTTP pada 64 pesan. Tanpa `app` |
| `sajikan_laya.py` | Langkah 6: `laya-serve` dengan checkpoint sendiri + pemanasan anti-SIGILL |
| `data/` | Keluaran langkah 1–3 (dibuat oleh skrip) |

## Langkah 1–3: dataset (lokal, dari `api/`)

```bash
.venv/Scripts/python -m eval.laya.buat_pesan                   # ±25 mnt, ±2.400 pesan
.venv/Scripts/python -m eval.laya.label_jev                    # ±20 mnt, ±$0,06 (JEV via gateway)
.venv/Scripts/python -m eval.laya.bangun_dataset               # detik
```

- Semua langkah bisa dilanjutkan bila terputus. Uji coba kecil:
  `buat_pesan --out eval/laya/data/pilot.jsonl --batas-tugas 1`.
- `buat_pesan` membaca potongan dokumen aktif dari database (8 dokumen per 2026-09-30) dan
  memakai `CHAT_MODEL` lewat `BASE_URL`. Bila gateway mengembalikan 503 (token codex
  dicabut), tugas yang gagal cukup dijalankan ulang.
- Setiap pesan membawa `niat` (kategori yang diminta dari LLM). `bangun_dataset` hanya
  melatih pesan yang label JEV-nya sama dengan niat; sisanya ke `data/tinjau.jsonl`.

**Meninjau `tinjau.jsonl`.** File ini juga daftar kesalahan JEV. Contoh 2026-09-30:
pertanyaan kode etik tentang intimidasi dilabeli `malicious` 0,93 dan pertanyaan tentang
persetujuan dilabeli `out_of_scope` 0,92, dan keduanya **diblokir JEV di produksi**.
Salin baris yang sudah ditinjau ke `data/keputusan.jsonl` dan tambahkan
`"keputusan": "<label>"` atau `"keputusan": "buang"`, lalu jalankan ulang
`bangun_dataset`. Baris itu dilatih dengan label hasil tinjauan (one-hot dihaluskan 0,9).
`keputusan.jsonl` per 2026-09-30 berisi 29 pesan akademik yang dilabeli lain oleh JEV.

`data/manifest.json` mencatat jumlah per label, `gerbang_sha256` (hash instruksi dan
kriteria gerbang), dan sha256 file. **Laya membaca teks instruksi dan kriteria sebagai
masukan**: bila `INSTRUCTIONS`/`CRITERIA` di `gate.py` berubah, bangun ulang dataset dan
latih ulang. Checkpoint menyimpan hash yang sama di `rl_agent_config.json` → `pandu`.

## Langkah 4: latih (GPU)

Server tidak bisa dipakai (tanpa GPU, CPU tanpa AVX), dan laptop kekurangan RAM.

**Kaggle (gratis):** unggah `data/kaggle.zip` sebagai Kaggle Dataset privat. File ini
dibuat oleh `bangun_dataset` dan berisi `latih_laya.py`, `evaluasi.py`, train, val, uji,
dan manifest. Buka `latih_kaggle.ipynb` di Kaggle (*File → Import Notebook*), tambahkan
dataset itu sebagai input, pilih GPU T4 dan Internet On, lalu Run All. Hasilnya
`laya-pandu-gerbang.tgz` di tab Output.

**Colab / mesin GPU lain:**

```bash
pip install "laya @ git+https://github.com/NandhaKishorM/laya@v0.3.22"
python latih_laya.py --data data --out laya-pandu-gerbang            # 4 epoch, batch 16
python evaluasi.py --uji data/uji.jsonl --model laya-pandu-gerbang --device cuda
```

Titik awalnya dipatok ke checkpoint yang sama dengan server (`convaiinnovations/laya`,
subfolder `multilingual`, revisi `55cf4c4e`). Loss RLCD dan hiperparameternya mengikuti
skrip resmi `laya_finetune_typed_decisions_mps.py`. Tambahannya:
- AMP fp16 di CUDA
- metrik validasi per epoch
- bobot terbaik menurut CE validasi, disimpan fp16 seperti checkpoint akhir
- temperature dikalibrasi pada set validasi dan dijepit ke rentang yang dipakai Agent (0,5–5)

Laporannya ada di `laporan_latih.json`. Uji asap di CPU (encoder dibekukan, 32 baris, 1
epoch) sudah lolos di server pada 2026-09-30. Rangkaiannya: simpan, `laya.load`, lalu
`sajikan_laya.py` + `evaluasi.py --url` untuk 64 pesan. Jalur GPU/AMP dan training encoder penuh
**belum pernah dijalankan**.

## Langkah 5: evaluasi

```bash
python evaluasi.py --uji data/uji.jsonl --model laya-pandu-gerbang    # checkpoint lokal
python evaluasi.py --uji data/uji.jsonl --model basis                 # pembanding zero-shot
LAYA_API_KEY=... python evaluasi.py --uji data/uji.jsonl --url http://127.0.0.1:8001/v1/systemone
```

**Syarat lulus:** 0 dari 38 akademik terblokir, dan paling sedikit 8 dari 9 pesan di luar
topik terblokir. Pembanding JEV v1: 1/76 akademik salah label, 0 terblokir, 18/18 di luar
topik terblokir.

Keputusan blokir dihitung dua kali, karena ada perbedaan skala keyakinan:
- `gate.parse_response` membaca field `confidence`.
- Pada Laya, untuk pertanyaan `choice`, `confidence` adalah **entropi ternormalisasi**
  (1 − H/log k), bukan probabilitas, dan tidak terkalibrasi.
- Yang terkalibrasi adalah `answer_confidence` (probabilitas label terpilih).

Sebelum Laya dipakai, `parse_response` perlu mengutamakan `answer_confidence` bila ada.
JEV tidak mengirim field itu, jadi perilaku JEV tidak berubah. Setelah itu ambang
dikalibrasi ulang.

## Langkah 6: sajikan di server (`/laya`)

`laya-serve` hanya mengenal checkpoint bawaan. `sajikan_laya.py` menjalankan server yang
sama (`laya.serve.main`), tapi nama `multilingual` diarahkan ke folder checkpoint. Gerbang
tetap mengirim `JEV_MODEL=multilingual`. Kalau `LAYA_PANDU_PATH` kosong, yang disajikan
adalah checkpoint asli.

**Anti-SIGILL.** Di CPU server, sekitar 13% proses torch mati pada prediksi pertamanya.
- Penyebabnya: MKL memilih kernel AVX-512 `mkl_vml_kernel_sCos_Z0HAynn`. Kernel itu
  dipanggil `torch.cos` di rotary embedding mmBERT. Instruksi yang crash adalah
  `vstmxcsr` di `libtorch_cpu.so`+`0xB724B91`.
- Hasil uji 30 proses baru per varian:
  - tanpa perbaikan: 4 crash
  - `MKL_ENABLE_INSTRUCTIONS=SSE4_2`: 5
  - `MKL_CBWR=SSE4_2`: 4
  - **`MKL_CBWR=COMPATIBLE`: 0**, dengan latensi yang sama
- `sajikan_laya.py` memasang `MKL_CBWR=COMPATIBLE` sebelum torch dimuat, lalu menjalankan
  satu prediksi pemanasan sebelum server menerima request. Kalau proses tetap crash,
  `restart: always` menyalakannya lagi. Rinciannya ada di `laya.md`.

Perubahan di `/laya` (belum diterapkan):

```bash
# dari laptop: salin checkpoint dan wrapper ke server, lalu di server:
sudo mkdir -p /laya/models && sudo tar xzf laya-pandu-gerbang.tgz -C /laya/models
sudo install -m 644 sajikan_laya.py /laya/sajikan_laya.py
```

```yaml
# /laya/docker-compose.yml, service laya
    command: ["python", "/opt/pandu/sajikan_laya.py"]
    volumes:
      - laya-data:/home/laya/.cache/huggingface
      - /laya/models/laya-pandu-gerbang:/models/laya-pandu-gerbang:ro
      - /laya/sajikan_laya.py:/opt/pandu/sajikan_laya.py:ro
    environment:
      LAYA_PANDU_PATH: /models/laya-pandu-gerbang
      MKL_CBWR: COMPATIBLE        # juga dipasang sajikan_laya.py sendiri
      # ... variabel lain tetap
```

Folder checkpoint harus bisa dibaca UID 10001. Setelah `sudo docker compose up -d`, uji
dengan `evaluasi.py --url`. Cara menyambungkannya ke chatbot-api (`JEV_URL`, `JEV_MODEL`,
`JEV_API_KEY`) ada di `laya.md`.
