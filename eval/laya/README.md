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
sudah dijalankan di Kaggle pada 2026-10-01 (putaran 1: 7 menit, validasi 96,6%, uji 57/64).

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

Perubahan di `/laya` (diterapkan user pada 2026-10-02 dengan checkpoint putaran 1):

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

## Putaran 2 (lanjutan fine-tuning)

Hasil putaran 1 (2026-10-01/02): validasi 96,6%, uji 57/64. Syarat lulus **belum** tercapai.
Daftar perbaikan beserta buktinya ada di `laya.md` → *Putaran 2: yang perlu diperbaiki*:
kode A1–A4, data B1–B5, evaluasi C1–C4, operasional D1–D4. Bagian ini berisi urutan
kerjanya. Tanda **(perlu dibuat)** berarti kode atau argumennya belum ada, jadi harus
ditulis dulu sebelum langkahnya bisa dijalankan.

### 0. Simpan hasil putaran 1

Bagian lokal sudah selesai pada 2026-10-02: `data/r1/` berisi train, val, uji, tinjau,
manifest, kaggle.zip, keputusan, dan hasil uji. Bagian server (D3) **belum**, dan baru
diperlukan sebelum langkah 5.

```bash
# lokal, dari api/
mkdir -p eval/laya/data/r1
cp eval/laya/data/{train.jsonl,val.jsonl,manifest.json,kaggle.zip,hasil_uji_pandu_20261002.jsonl} eval/laya/data/r1/
# server (D3): cadangan untuk rollback, ±650 MB
sudo cp -a /laya/models/laya-pandu-gerbang /laya/models/laya-pandu-gerbang-r1
```

### 1. Perbaiki kode gerbang (A1–A4, tanpa latih ulang)

**Selesai 2026-10-02** (hasilnya di `laya.md` → *Hasil A1/A2/A4*):

1. `gate.parse_response` mengambil keyakinan dari `answer_confidence`. Kalau field itu
   tidak ada (JEV), dipakai `confidence`.
2. `GatePolicy.nonsense_threshold` dan `JEV_NONSENSE_THRESHOLD`. Kalau kosong, nilainya
   sama dengan `JEV_BLOCK_THRESHOLD`, jadi JEV tidak berubah. Di `api/.env` lokal dipasang
   0,95 untuk sementara.
3. `JEV_URL` tanpa `http://`/`https://` ditolak saat start.
4. Ditambah 11 test di `tests/unit/test_gate.py`; suite 1714 lolos, dan 16 kegagalannya
   sama dengan baseline. Langkah ini tidak mengubah `gerbang_sha256`, jadi checkpoint
   putaran 1 masih sah.

Setelah langkah ini, gerbang lokal sudah lebih baik walau model belum dilatih ulang
(simulasinya di `laya.md`). Ambangnya baru dipilih di langkah 7.

### 2. Siapkan set kalibrasi dan perluas set uji (C2, C3)

**Selesai 2026-10-02.** Hasil dan baseline r1-nya ada di `laya.md` → *Hasil langkah 0 dan 2*.

- `set_uji.py`: `SET` (64 pesan, asal `r1`) tidak diubah, ditambah `SET_TAMBAHAN`
  (36 pesan, asal `r2`, bentuk tuple `(label, unit, pesan, riwayat)`). `semua()` mengembalikan
  keduanya. Totalnya 100 pesan, 20 di antaranya punya riwayat.
- `set_kalibrasi.py`: 60 pesan, termasuk 11 pertanyaan nyata dari `messages` beserta
  riwayat aslinya.
- `bangun_dataset.py` membuang pesan latih yang sama atau mirip dengan kedua set, lalu
  menulis `uji.jsonl` dan `kalibrasi.jsonl` (dengan `riwayat` dan `asal`).
  `kalibrasi.jsonl` ikut masuk `kaggle.zip`.
- `evaluasi.py` menampilkan ringkasan per asal. Syarat lulusnya ≥ 90% di luar topik
  terblokir, dibulatkan ke bawah.
- Pesan baru di kedua set tidak boleh sama atau mirip data latih. Cek ini sudah dijalankan
  terhadap r1: satu pesan ("ppppp") bocor dan sudah diganti.
- **Jangan ubah isi set uji** setelah ada angka putaran 2. Kalau perlu kasus baru, tambahkan
  dengan asal baru (`r3`), supaya angka lama tetap bisa dibandingkan.

Mengukur checkpoint yang sedang disajikan pada kedua set (lewat tunnel, dari `api/`):

```bash
export LAYA_API_KEY="$(grep '^JEV_API_KEY=' .env | cut -d= -f2-)"
.venv/Scripts/python eval/laya/evaluasi.py --uji eval/laya/data/kalibrasi.jsonl \
  --url http://127.0.0.1:8001/v1/systemone --out eval/laya/data/hasil_kalibrasi_pandu_r1.jsonl
```

### 3. Tambah data (B1–B4)

**(perlu dibuat)** Tambahkan kelompok baru di `buat_pesan.py`, beserta argumen `--kelompok`
supaya bisa hanya menjalankan kelompok baru:

| Kelompok | Isi | Cara |
|---|---|---|
| `pendek` (B1) | ±200 akronim/kata kunci akademik 1–3 kata: ukm, krs, ukt, kip, skp, sks, ipk, mbkm, toeic, ccna, va, … ditambah "brp", "?", "gimana", "dong" | Program, dari `glossary.KELOMPOK` dan `rule_gate._ISTILAH_KAMPUS`, `niat=academic` |
| `riwayat` (B2) | 1–3 giliran riwayat dengan jawaban asisten panjang bergaya PANDU (±300–700 karakter, sitasi, langkah bernomor), untuk semua label. Termasuk pertanyaan lanjutan pendek → academic, pindah ke luar tema → out_of_scope, dan manipulasi di tengah percakapan → malicious | LLM dari `ambil_potongan`. Target: ≥ 30% baris tiap label punya riwayat |
| tambahan `LUAR_TEMA` (B3) | Mengerjakan tugas/PR, materi kuliah, pengetahuan umum, kampus lain | Tema baru di daftar yang sudah ada |
| tambahan `SULIT_TEMA` (B4) | Merek bank/sertifikasi yang memang ada di dokumen (ATM Bersama, BRI → VA BNI, CCNA, IC3) | Tema baru di daftar yang sudah ada |

Jangan membuat ulang pesan yang persis sama dengan set uji. `bangun_dataset` memang akan
membuangnya, tapi kalau itu terjadi, jumlah datanya jadi kurang.

```bash
# dari api/. File putaran 1 tetap utuh.
.venv/Scripts/python -m eval.laya.buat_pesan --out eval/laya/data/pesan_r2.jsonl --kelompok pendek,riwayat,luar,sulit
.venv/Scripts/python -m eval.laya.label_jev --pesan eval/laya/data/pesan_r2.jsonl --out eval/laya/data/label_jev_r2.jsonl
# (perlu dibuat) --pesan/--label bisa diulang supaya data putaran 1 dan 2 digabung
.venv/Scripts/python -m eval.laya.bangun_dataset \
  --pesan eval/laya/data/pesan.jsonl --pesan eval/laya/data/pesan_r2.jsonl \
  --label eval/laya/data/label_jev.jsonl --label eval/laya/data/label_jev_r2.jsonl
```

- `label_jev` memakai body dari `gate.build_request`, jadi riwayat ikut dinilai JEV.
- Tinjau `tinjau.jsonl` lalu isi `keputusan.jsonl`, sama seperti putaran 1. Perhatikan
  terutama akronim pendek yang dilabeli JEV sebagai `nonsense`.
- Cek `manifest.json` sebelum lanjut: jumlah akademik 1–3 kata ≥ 200, dan baris yang punya
  riwayat ≥ 30% di tiap label.
- `kalibrasi.jsonl` sudah ikut masuk `kaggle.zip` (sejak langkah 2).
- Tambahan dari hasil langkah 2:
  - `riwayat` wajib memuat kasus pindah topik, manipulasi, dan basa-basi setelah riwayat
    akademik. Di r1, 72% baris yang punya riwayat berlabel academic, sehingga model
    melabeli "abaikan dokumen resmi…" dan "mantap, jelas banget…" sebagai academic
    dengan p 0,95–0,96.
  - `LUAR_TEMA` perlu ditambah tema kampus lain, pendaftaran perguruan tinggi lain, dan
    karier (CV, CPNS).

### 4. Latih di Kaggle

1. Buka Kaggle → *Datasets* → dataset putaran 1 → **New Version**, lalu unggah `kaggle.zip`
   yang baru. Pakai versi baru, bukan dataset baru, supaya input notebook tidak perlu diubah.
2. Di sel latih `latih_kaggle.ipynb`, ganti `--epochs 4` menjadi `--epochs 6` (B5).
   **(perlu dibuat)** Tambahkan sel evaluasi untuk `data/kalibrasi.jsonl`.
3. Jalankan *Save Version → Save & Run All*. Setelah selesai, unduh
   `laya-pandu-gerbang.tgz`, `laporan_latih.json`, dan keluaran sel-sel evaluasi.
4. Bandingkan per epoch dengan putaran 1 (`r1/`). **(perlu dibuat)** Tambahkan juga
   akurasi per `sumber` dan per panjang pesan (C4).

### 5. Pasang di server (dengan jalan mundur)

```bash
scp laya-pandu-gerbang.tgz embed@100.111.178.48:~
# di server (cadangan r1 sudah dibuat di langkah 0)
sudo rm -rf /laya/models/laya-pandu-gerbang
sudo tar xzf ~/laya-pandu-gerbang.tgz -C /laya/models && sudo chmod -R a+rX /laya/models
cd /laya && sudo docker compose up -d --force-recreate
sudo docker compose logs | grep "pemanasan lolos"
```

Compose tidak perlu diubah, karena path folder checkpoint-nya tetap. Untuk **rollback**,
kembalikan folder dari `laya-pandu-gerbang-r1`, lalu jalankan
`sudo docker compose up -d --force-recreate`.

### 6. Tunnel dari laptop (D1)

```bash
ssh -N -L 8001:127.0.0.1:8001 embed@100.111.178.48   # biarkan terbuka
```

`api/.env` lokal: `JEV_URL=http://127.0.0.1:8001/v1/systemone`, `JEV_MODEL=multilingual`,
`JEV_API_KEY` = `LAYA_API_KEY` server. Setelah `.env` diubah, jalankan `touch app/config.py`.

### 7. Pilih ambang di set kalibrasi, lalu uji sekali

```bash
# dari api/. LAYA_API_KEY = JEV_API_KEY di api/.env
# (perlu dibuat) --pipeline: rule_gate + smalltalk dulu (C1); --ambang-nonsense
.venv/Scripts/python eval/laya/evaluasi.py --uji eval/laya/data/kalibrasi.jsonl \
  --url http://127.0.0.1:8001/v1/systemone --pipeline
```

- Pilih ambang `out_of_scope` paling rendah yang masih memblokir ≥ 90% pesan di luar topik
  tanpa memblokir satu pun pesan akademik. Beri jarak ±0,1 dari p tertinggi pesan akademik.
  Ambang `nonsense` dipilih dengan cara yang sama.
- Baru setelah itu jalankan set uji, **sekali saja**:

```bash
.venv/Scripts/python eval/laya/evaluasi.py --uji eval/laya/data/uji.jsonl \
  --url http://127.0.0.1:8001/v1/systemone --pipeline \
  --ambang-luar <hasil kalibrasi> --ambang-nonsense <hasil kalibrasi> \
  --out eval/laya/data/hasil_uji_pandu_r2.jsonl
```

- Isi ambang yang sama ke `api/.env` (`JEV_OUT_OF_SCOPE_THRESHOLD`, `JEV_NONSENSE_THRESHOLD`).

### 8. Uji widget

Ikuti `browseract.md`: pakai `chrome-live` dengan tunnel aktif. Kasus yang diuji:
- akronim pendek ("ukm", "krs?", "toeic brp")
- pertanyaan lanjutan ("kalau lewat teller?")
- pindah topik ke luar tema setelah percakapan akademik
- pesan di luar topik yang mirip akademik ("kerjakan tugas kalkulus saya")

Pesan di luar topik harus berhenti di node `jev_gate` (`outcome=rejected` tanpa node
`generate` di `log/app.db`). Setelah selesai, hapus datanya dengan `--awalan uji-`.

### 9. Catat dan putuskan

- Tulis hasilnya di `laya.md` sebagai *Hasil putaran 2*, lalu tandai butir A–D yang sudah selesai.
- Kalau lulus, ikuti *Menyambungkan ke chatbot-api* di `laya.md`. Kalau tidak, lakukan
  rollback (langkah 5) dan catat penyebabnya.
