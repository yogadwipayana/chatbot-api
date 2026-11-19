# Melatih Laya di Kaggle: pelatihan kedua (putaran 3, T70)

Panduan langkah demi langkah untuk melatih ulang model gerbang Laya (`laya-pandu-gerbang`)
di Kaggle, mengukurnya, lalu memasangnya di server. Dokumen ini khusus untuk **pelatihan
kedua**, yaitu putaran 3 (asal data `r3`). Pelatihan pertama (putaran 1, 2026-10-01) dan
latar belakang Laya ada di `docs/laya.md`. Rincian skrip ada di `eval/laya/README.md`.

**Urutannya: ukur di Kaggle dulu, baru pasang.**

```
dataset (lokal) → unggah → latih + ukur + putuskan (Kaggle)
  → LULUS?  tidak → perbaiki data, latih lagi; produksi tetap r1
            ya    → pasang di server → cek kesesuaian server = Kaggle → cek produksi
```

Checkpoint yang belum terbukti tidak pernah melayani mahasiswa. Pemeriksaan di server
setelah pemasangan hanya memastikan hasilnya sama dengan di Kaggle, bukan mengukur ulang.

| | Pelatihan pertama (r1) | Pelatihan kedua (r3) |
|---|---|---|
| Tanggal data | 2026-09-30 | 2026-11-19 |
| Alasan | Ganti JEV dengan model lokal | T70: r1 memblokir pertanyaan nilai mahasiswa |
| Data latih / validasi | 2.109 / 235 | 2.461 / 274 (r1 + 395 pesan baru) |
| Set uji / kalibrasi | 64, lalu 100 (r2) / 60 | 124 / 72 |
| `gerbang_sha256` | `48a9d7cd…` | `48a9d7cd…` (sama) |
| Berkas Kaggle | `eval/laya/data/kaggle.zip` | `eval/laya/data/r3/kaggle.zip` |

## 1. Kenapa dilatih ulang

Sejak 2026-11-07 chatbot bisa menjawab data akademik milik penanya sendiri lewat tool SADS:
nilai, IPK/SKS, jadwal, KRS, tagihan, dan dosen. Pertanyaan seperti ini tidak ada di data
latih r1. Pada 2026-11-18 Laya r1 di produksi memblokir "nilaiku berapa yang paling tinggi
dan berapa yang peling rendah" sebagai `out_of_scope` dengan keyakinan 0,81. Ambangnya 0,7,
jadi mahasiswa mendapat penolakan "di luar topik" untuk pertanyaan yang sebenarnya sah.

**CRITERIA gerbang tidak diubah.** Ada dua alasan:
- Guru pelabel (JEV) sudah melabeli pertanyaan nilai/IPK sebagai `academic` (0,88–1,0)
  dengan teks yang sekarang.
- `build_sequence` Laya memotong setiap opsi kriteria di 48 token, jadi kalimat tambahan di
  ujung kriteria tidak akan terbaca model.

Dengan teks yang sama, `gerbang_sha256` tidak berubah, data r1 tetap sah, dan `gate.py` tidak
perlu disentuh saat checkpoint baru dipasang. Yang ditambah hanya **data**.

**Titik awal pelatihan: model dasar, bukan r1.** Putaran 3 **tidak** melanjutkan checkpoint
r1 yang sedang disajikan server (`/laya/models/laya-pandu-gerbang`). `latih_laya.py` selalu
mengunduh model dasar publik `convaiinnovations/laya` (subfolder `multilingual`, revisi
`55cf4c4e`), yaitu model yang disajikan server sebelum fine-tuning, lalu melatihnya dengan
**seluruh** data: r1 + 395 pesan r3. Alasannya:
- **Perbandingan bersih.** Model dasar, hiperparameter, seed, dan jumlah epoch sama dengan
  r1; yang berbeda hanya datanya. Hasilnya tidak identik bit demi bit dengan r1, karena data
  yang bertambah ikut mengubah urutan acak.
- **Mencegah lupa.** Melanjutkan r1 hanya dengan data baru, yang hampir semuanya akademik,
  bisa menggeser pola lama, misalnya meloloskan pesan di luar topik.
- **Praktis.** Checkpoint r1 (±650 MB) hanya ada di server, sedangkan latih dari model dasar
  cukup ±10–15 menit.

`"fine_tuned_from_checkpoint": true` di `rl_agent_config.json` hasil latih hanya berarti
model dilatih dari checkpoint yang sudah ada, bukan dari nol. Bukan berarti dari r1. Titik
awal yang sebenarnya tercatat di `training.basis` dan di `laporan_latih.json` → `basis`
(`convaiinnovations/laya/multilingual@55cf4c4e…`).

## 2. Baseline: r1 yang sedang aktif

Diukur 2026-11-19 lewat HTTP ke server (`--ambang 0.95 --ambang-luar 0.7`, ukuran
`answer_confidence` seperti `gate.py`). Hasil per pesan ada di
`eval/laya/data/r3/hasil_r1_uji.jsonl` dan `hasil_r1_kalibrasi.jsonl` (dengan `.log`-nya).

| Set | Akurasi | Akademik terblokir | Di luar topik terblokir |
|---|---|---|---|
| uji, semua (124) | 98/124 (79%) | **4/77** | 19/24 |
| uji, asal r1 (64) | 57/64 | 0/38 | 9/9 |
| uji, asal r2 (36) | 26/36 | 0/20 | 7/11 |
| uji, asal r3 (24) | 15/24 | **4/19** | 3/4 |
| kalibrasi, semua (72) | 66/72 | 0/35 | 17/21 |
| kalibrasi-r3 (12) | 12/12 | 0/9 | 3/3 |

Keempat pesan akademik r3 yang terblokir adalah pesan nyata dari log produksi:
- nilai tertinggi/terendah
- "tunjukan semua nilaiku"
- "jadwalku sekarang apa aja ya?"
- "kalau dari progres ku apakah aku sudah bisa ambil sempro"

Empat lainnya berlabel `out_of_scope` tetapi masih di bawah ambang (0,53–0,68).

## 3. Dataset (sudah selesai, lokal)

Dijalankan dari `api/` pada 2026-11-19. Langkah ini tidak perlu diulang kecuali datanya
mau diubah.

```bash
# 1. buat pesan (17 tugas LLM lewat gateway, ±3 menit)
.venv/Scripts/python -m eval.laya.buat_pesan --out eval/laya/data/pesan_r3.jsonl \
  --kelompok pribadi,pribadi-lanjutan,pribadi-luar

# 2. label dengan guru JEV lewat gateway (395 pesan, ±3,5 menit, < $0,01)
.venv/Scripts/python -m eval.laya.label_jev --pesan eval/laya/data/pesan_r3.jsonl \
  --out eval/laya/data/label_jev_r3.jsonl

# 3. gabungkan r1 + r3 ke folder baru; data putaran 1 di data/ tidak disentuh
.venv/Scripts/python -m eval.laya.bangun_dataset --dir eval/laya/data/r3 \
  --pesan eval/laya/data/pesan.jsonl --pesan eval/laya/data/pesan_r3.jsonl \
  --label eval/laya/data/label_jev.jsonl --label eval/laya/data/label_jev_r3.jsonl \
  --keputusan eval/laya/data/keputusan.jsonl
```

**Isi data baru** (kelompok di `eval/laya/buat_pesan.py`):

| Kelompok | Jumlah | Niat | Isi |
|---|---|---|---|
| `pribadi` | 300 | academic | 6 tema × 2 variasi: nilai, IPK/SKS/progres, jadwal, KRS/kurikulum/prasyarat, tagihan, dosen. Gaya orang pertama (saya, aku, -ku, gw), banyak yang pendek dan salah ketik |
| `pribadi-lanjutan` | 45 | academic | Pesan susulan pendek yang hanya bermakna dengan riwayat ("kalau yg terendah?", "semester 3 gmn") |
| `pribadi-luar` | 50 | out_of_scope | Pembanding dengan kata yang sama: kurs, nilai saham, tagihan listrik/kartu kredit, jadwal kereta/bola, IPK untuk lowongan kerja, kampus lain |

- Unit acak memuat nama lama dan nama sekarang (`AKADEMIK`, `KEUANGAN`, `BAAK`,
  `Keuangan`, …), karena unit diganti nama pada 2026-11-18 dan `topik_dipilih` ikut
  terbaca Laya.
- **Kesepakatan JEV dengan niat:** 299/300, 45/45, 40/50.
- **11 pesan yang tidak sepakat** ditinjau dan dicatat di `eval/laya/data/keputusan.jsonl`,
  dengan alasan di tiap baris:
  - 1 tetap `academic`: "tiga nilai terbsar aku apa aja?"
  - 10 tetap `out_of_scope`: tagihan kartu kredit/listrik, jadwal kereta, IPK untuk
    lowongan, sistem nilai kampus lain (sesuai `docs/laya.md` B3).
  - Tinjau ulang bila perlu. Kalau ada yang diubah, jalankan ulang langkah 3.
- **Set uji dan kalibrasi baru:**
  - `set_uji.SET_R3`: 24 kasus asal `r3`, yaitu 19 pertanyaan akademik nyata dari log
    produksi 2026-11-12..18 (3 dengan riwayat), 1 malicious (IPK teman dengan NIM
    fiktif), dan 4 out_of_scope.
  - `set_kalibrasi.SET_R3`: 12 kasus buatan tangan, asal `kalibrasi-r3`.
  - `bangun_dataset` membuang pesan latih yang sama atau mirip dengan kedua set (10 baris).

**Hasil di `eval/laya/data/r3/`:** `train.jsonl` (2.461), `val.jsonl` (274), `uji.jsonl`
(124), `kalibrasi.jsonl` (72), `tinjau.jsonl`, `manifest.json`, dan **`kaggle.zip`**.
`kaggle.zip` berisi `latih_laya.py`, `evaluasi.py`, keempat JSONL, dan manifest.

| Label | train | val |
|---|---|---|
| academic | 1.533 | 170 |
| out_of_scope | 501 | 56 |
| malicious | 186 | 21 |
| smalltalk | 145 | 16 |
| nonsense | 96 | 11 |

Butir B1–B4 putaran 2 (akronim pendek, riwayat realistis, tema di luar topik tambahan,
merek bank) **belum** termasuk. Tujuannya supaya perbedaan r1 → r3 bisa dikaitkan dengan
data pribadi saja.

## 4. Persiapan akun Kaggle

- Akun Kaggle harus sudah **terverifikasi nomor ponsel**. Tanpa itu, GPU dan *Internet* di
  notebook tidak bisa dinyalakan.
- Kuota GPU gratis ±30 jam per minggu. Satu kali latih memakai ±10–15 menit.
- Notebook butuh *Internet On* untuk `pip install laya@v0.3.22` dan untuk mengunduh
  model dasar `convaiinnovations/laya` (subfolder `multilingual`, revisi `55cf4c4e`) dari
  Hugging Face. Itu titik awal pelatihan (lihat langkah 1); checkpoint r1 tidak perlu
  diunggah ke Kaggle.

## 5. Unggah dataset

Pakai dataset Kaggle yang sama dengan putaran 1, sebagai versi baru. Dengan begitu input
notebook tidak perlu diubah.

1. Kaggle → **Datasets** → *Your Work* → dataset putaran 1 (dataset privat yang dulu
   dibuat dari `eval/laya/data/kaggle.zip`).
2. Klik **New Version**.
3. Hapus berkas lama, lalu unggah `api/eval/laya/data/r3/kaggle.zip` (±8 MB setelah
   diekstrak). Kaggle mengekstrak zip secara otomatis.
4. Isi *Version notes*, misalnya `r3 (T70): r1 + 395 pesan data pribadi, train 2461 /
   val 274, gerbang 48a9d7cd`.
5. **Create**, lalu tunggu sampai status versinya selesai diproses.

Kalau dataset putaran 1 sudah tidak ada, buat dataset privat baru dari zip yang sama. Di
langkah 6, tambahkan dataset baru itu sebagai input. Notebook mencari `train.jsonl` di
mana saja di bawah `/kaggle/input`, jadi nama dataset tidak berpengaruh.

## 6. Latih dan ukur di Kaggle

1. Buka notebook putaran 1, atau impor `api/eval/laya/latih_kaggle.ipynb` lewat
   *File → Import Notebook*. Notebook putaran 1 harus diganti dengan versi ini, karena
   versi lama belum punya sel ukur dan sel keputusan.
2. Panel kanan:
   - **Input**: dataset dari langkah 5. Pastikan yang terpasang versi terbaru; kalau
     notebook masih memakai versi lama, pilih *Update to latest version*.
   - **Accelerator**: GPU T4 x1 (atau P100).
   - **Internet**: On.
3. Jangan ubah hiperparameter. Sel latih memakai `--epochs 4 --micro-batch 16`, sama
   dengan r1, supaya perbandingannya bersih. `--epochs 6` (B5 di `docs/laya.md`) dicoba
   sebagai eksperimen terpisah, bukan di pelatihan ini.
4. **Save Version → Save & Run All (Commit)**. Notebook berjalan di latar dan tidak
   berhenti walau tab ditutup.

**Isi sel notebook:**

| Sel | Isi | Yang dicek |
|---|---|---|
| 1 | `nvidia-smi`, `pip install laya@v0.3.22` | `cuda True` |
| 2 | Salin `latih_laya.py`, `evaluasi.py`, train, val, uji, kalibrasi, dan manifest ke `/kaggle/working` | Manifest tercetak dengan `train` 2461 dan `gerbang_sha256` `48a9d7cd…`; berhenti dengan galat bila `kalibrasi.jsonl` tidak ada |
| 4 | Latih 4 epoch | Metrik validasi per epoch, `epoch terbaik`, `temperature choice` |
| 6 | Ukur `kalibrasi.jsonl` dengan ambang produksi → `hasil_kalibrasi.jsonl` | Baris `answer_confidence (gate.py)` |
| 7 | Ukur `uji.jsonl` **sekali**, dengan ambang yang sama → `hasil_uji.jsonl` | Ringkasan per asal (r1, r2, r3) |
| 9 | Tabel keputusan dibanding r1 → `keputusan.json` | `KEPUTUSAN: LULUS` atau `TIDAK LULUS`, plus daftar pesan akademik yang terblokir |
| 11 | Checkpoint dasar (zero-shot), hanya sebagai konteks | – |
| 12 | Cetak `laporan_latih.json`, kemas `laya-pandu-gerbang.tgz` | Ukuran ±600 MB |

**Ambang pengukuran:** `--ambang 0.95 --ambang-luar 0.7`, sama dengan produksi dan dengan
baseline r1 di langkah 2.
- out_of_scope 0,7 = `LAYA_OUT_OF_SCOPE_THRESHOLD`.
- 0,95 = `LAYA_NONSENSE_THRESHOLD`. `evaluasi.py` hanya punya satu ambang untuk label
  selain out_of_scope, jadi 0,95 ikut dipakai untuk smalltalk dan malicious.
- Keputusan blokir dibaca dari `answer_confidence`, seperti `gate.py`.

## 7. Baca keputusan

Sel 9 membandingkan checkpoint baru dengan baseline r1 memakai syarat berikut. Angkanya
berlaku untuk dataset putaran 3 (uji 124, kalibrasi 72); kalau dataset berubah, perbarui
angka di sel 9 dan di tabel ini.

| Ukuran | Baseline r1 | Syarat |
|---|---|---|
| Akademik terblokir, uji semua | 4/77 | **0/77** |
| Akademik terblokir, uji asal r3 | 4/19 | **0/19** |
| Akademik terblokir, kalibrasi | 0/35 | 0/35 |
| Di luar topik terblokir, uji semua | 19/24 | ≥ 19/24 (tidak boleh turun) |
| Di luar topik terblokir, kalibrasi | 17/21 | ≥ 17/21 |
| Akurasi uji asal r1 | 57/64 | ≥ 55/64 (tidak boleh turun jauh) |

- **LULUS**: lanjut ke langkah 8 dan 9.
- **TIDAK LULUS**: jangan pasang; produksi tetap memakai r1.
  - Sel 9 mendaftar pesan akademik yang terblokir. Cari polanya di data latih (tidak
    terwakili, atau label JEV keliru), perbaiki datanya, bangun ulang dataset, unggah
    sebagai versi baru, lalu latih lagi.
  - Set uji ikut terukur di setiap kali latih. Catat berapa kali set uji sudah dilihat
    di `docs/laya.md`. Pilih perbaikan berdasarkan kalibrasi dan validasi, jangan
    berdasarkan pesan tertentu di set uji, supaya angka uji tetap jujur untuk skripsi.
- Ambang produksi tidak diubah di putaran ini. Kalau kalibrasi menunjukkan ambang perlu
  digeser, pilih ambangnya **dari kalibrasi saja**. Mengubah `LAYA_*_THRESHOLD` di
  `/chatbot/api/.env` perlu persetujuan user dan recreate container api.

## 8. Unduh dan periksa hasil

Dari halaman versi notebook → tab **Output**, unduh:
- `laya-pandu-gerbang.tgz`: checkpoint (`model.safetensors` fp16, `encoder/`,
  `tokenizer/`, `rl_agent_config.json`, `laporan_latih.json`)
- `hasil_kalibrasi.jsonl`, `hasil_uji.jsonl`, `keputusan.json`

Simpan hasilnya di `api/eval/laya/data/r3/` supaya terarsip bersama datanya, dengan nama
`hasil_r3_kalibrasi_kaggle.jsonl`, `hasil_r3_uji_kaggle.jsonl`, `keputusan_kaggle.json`,
dan `laporan_latih_kaggle.json` (diambil dari dalam tgz).

**Periksa `laporan_latih.json`** dan bandingkan dengan r1 (4 epoch, 7 menit, epoch terbaik
4, temperature `choice` 1,148, validasi 96,6%):
- `epoch_terbaik` dan tren CE validasi per epoch. Kalau masih turun di epoch 4, catat
  sebagai alasan untuk mencoba 6 epoch nanti.
- `temperature_choice` di rentang 0,5–5.
- `val_terkalibrasi`: akurasi validasi. Angka ini sintetis dan cenderung optimistis
  (`docs/laya.md` C4).

**Periksa `rl_agent_config.json` → `pandu`:**
- `gerbang_sha256` = `48a9d7cd17523198…`. Kalau berbeda, checkpoint tidak cocok dengan
  `gate.py`.
- `data_sha256["train.jsonl"]` = `541862efb35cd228…` (`data/r3/manifest.json`). Ini bukti
  bahwa yang dilatih memang data r3, bukan versi dataset lama.
- `training.n_train` = 2461.

## 9. Pasang di server (`/laya`), hanya bila LULUS

Server: `embed@100.111.178.48`, folder `/laya`. Pemilik berkasnya root, jadi perlu
`sudo`. Container `laya` membaca checkpoint dari `/laya/models/laya-pandu-gerbang`
(read-only), lewat `sajikan_laya.py` dengan `MKL_CBWR=COMPATIBLE`.

```bash
# dari laptop
scp laya-pandu-gerbang.tgz embed@100.111.178.48:~

# di server
ls /laya/models
# Cadangan r1 untuk rollback. Sekali saja: lewati bila laya-pandu-gerbang-r1 sudah ada.
sudo cp -a /laya/models/laya-pandu-gerbang /laya/models/laya-pandu-gerbang-r1

sudo rm -rf /laya/models/laya-pandu-gerbang
sudo tar xzf ~/laya-pandu-gerbang.tgz -C /laya/models
sudo chmod -R a+rX /laya/models/laya-pandu-gerbang          # dibaca UID 10001
grep -o '"n_train": [0-9]*' /laya/models/laya-pandu-gerbang/rl_agent_config.json   # 2461

cd /laya && sudo docker compose up -d --force-recreate
sudo docker compose logs --tail 50 | grep "pemanasan lolos"
```

- Compose tidak perlu diubah, karena path checkpoint-nya tetap.
- Setiap checkpoint ±650 MB. Cek `df -h /` sebelum menyalin; disk root 61 GB.
- Selama container dibuat ulang (±30 detik), gerbang **fail-open**: pesan tetap diteruskan
  ke retrieval tanpa diblokir. Lakukan di luar jam ramai.
- Chatbot-api tidak perlu restart. `LAYA_URL`, `LAYA_MODEL=multilingual`, dan ambangnya
  tetap.

## 10. Cek kesesuaian server dengan Kaggle

Di Kaggle model berjalan di GPU, sedangkan di server di CPU (fp16 dimuat ke CPU, dengan
`MKL_CBWR=COMPATIBLE`). Probabilitasnya bisa berbeda sedikit. Langkah ini memastikan
**keputusannya** tetap sama, dan tidak mengukur ulang. Yang dijalankan hanya set
kalibrasi; set uji tidak dijalankan lagi.

Dari `api/` di laptop. Pakai alamat Tailscale, karena domain publik
`https://laya.dwipa.my.id` menolak User-Agent `urllib` milik `evaluasi.py` (HTTP 403 dari
Cloudflare).

```bash
export LAYA_API_KEY="$(grep '^LAYA_API_KEY=' .env | cut -d= -f2- | tr -d '\r')"
.venv/Scripts/python eval/laya/evaluasi.py --uji eval/laya/data/r3/kalibrasi.jsonl \
  --url http://100.111.178.48:8001/v1/systemone --ambang 0.95 --ambang-luar 0.7 \
  --out eval/laya/data/r3/hasil_r3_kalibrasi_server.jsonl

.venv/Scripts/python - <<'EOF'
import json
baca = lambda p: [json.loads(b) for b in open(p, encoding="utf-8") if b.strip()]
k = baca("eval/laya/data/r3/hasil_r3_kalibrasi_kaggle.jsonl")
s = baca("eval/laya/data/r3/hasil_r3_kalibrasi_server.jsonl")
beda = [(a["pesan"][:60], a["choice"], b["choice"], a["blok_answer"], b["blok_answer"])
        for a, b in zip(k, s) if (a["choice"], a["blok_answer"]) != (b["choice"], b["blok_answer"])]
selisih = max(abs(a["answer_confidence"] - b["answer_confidence"]) for a, b in zip(k, s))
print(f"{len(k)} pesan, {len(beda)} keputusan berbeda, selisih p maks {selisih:.3f}")
for x in beda:
    print("  ", x)
EOF
```

- **0 keputusan berbeda**: checkpoint di server sama dengan yang diukur di Kaggle. Lanjut
  ke langkah 11.
- **Ada yang berbeda**, terutama pesan akademik yang di server terblokir: lakukan rollback
  (langkah 12) dan selidiki. Kemungkinan penyebabnya: tgz atau versi dataset yang salah
  (cek langkah 8), atau perbedaan numerik CPU/GPU di dekat ambang.

## 11. Cek di produksi

1. Admin → **Uji coba jawaban**, unit `AKADEMIK`. Tanyakan "nilaiku berapa yang paling
   tinggi dan berapa yang peling rendah" dan "jadwalku sekarang apa aja ya?". Baris
   "Gerbang Laya" harus `akademik … diteruskan`.
2. Tanyakan juga satu pesan di luar topik ("resep rendang", "nilai tukar rupiah hari ini").
   Pesan ini harus berhenti di Gerbang Laya.
3. Admin → **Log → Graf** pada giliran tadi. Langkah *Gerbang Laya*, tab Panggilan,
   `laya_classify`: cek label dan probabilitasnya.
4. Selama beberapa hari, pantau Log → Giliran chat dengan filter Hasil "Di luar topik" dan
   Jalur. Pertanyaan akademik yang berhenti di `laya_gate` berarti ada regresi.

## 12. Rollback

```bash
# di server
sudo rm -rf /laya/models/laya-pandu-gerbang
sudo cp -a /laya/models/laya-pandu-gerbang-r1 /laya/models/laya-pandu-gerbang
cd /laya && sudo docker compose up -d --force-recreate
sudo docker compose logs --tail 50 | grep "pemanasan lolos"
```

Lakukan rollback bila langkah 10 menemukan keputusan yang berbeda, atau bila langkah 11
menemukan pertanyaan akademik yang diblokir. Catat penyebabnya.

## 13. Catat hasil

- `docs/laya.md`: bagian *Hasil putaran 3*, berisi tabel langkah 7 (r1 vs r3, dari
  `keputusan_kaggle.json`), hasil langkah 10, ringkasan `laporan_latih.json`, berapa kali
  set uji dilihat, dan keputusan pasang/rollback.
- `eval/laya/README.md` → *Putaran 3*: tandai langkah yang sudah selesai.
- `browseract.md` §5: tutup T70 setelah uji ulang di produksi lolos.
- Simpan semua berkas hasil (`hasil_r3_*`, `keputusan_kaggle.json`,
  `laporan_latih_kaggle.json`) di `eval/laya/data/r3/`, supaya angka di skripsi bisa
  direproduksi.

## Masalah yang pernah muncul

| Gejala | Penyebab | Penanganan |
|---|---|---|
| `evaluasi.py --url https://laya.dwipa.my.id/…` → HTTP 403 | Cloudflare menolak User-Agent `Python-urllib` | Pakai `http://100.111.178.48:8001/v1/systemone` (Tailscale) |
| `label_jev` → 401 "User not found" | Kredensial OpenRouter di gateway kosong atau kedaluwarsa | Perbarui kredensial OpenRouter di gateway (2026-11-19) |
| `buat_pesan`/`label_jev` → 503 `token_revoked` | Token codex gateway dicabut | Login ulang provider codex di gateway, lalu jalankan ulang (tugas yang selesai dilewati) |
| Proses torch di server mati (SIGILL) pada prediksi pertama | MKL memilih kernel AVX-512 di CPU tanpa AVX | `MKL_CBWR=COMPATIBLE` (sudah di compose dan `sajikan_laya.py`) |
| Kaggle tidak menemukan `train.jsonl` | Dataset belum ditambahkan sebagai input, atau masih versi lama | *Add Input* / *Update to latest version* di panel kanan |
| OOM saat menyimpan bobot di mesin kecil | Bobot fp32 | `latih_laya.py` sudah menyimpan fp16; di CPU pakai `--freeze-encoder` hanya untuk uji asap |

## Berkas terkait

- `eval/laya/buat_pesan.py`: kelompok `pribadi`, `pribadi-lanjutan`, `pribadi-luar`, argumen `--kelompok`
- `eval/laya/label_jev.py`: guru JEV lewat `<BASE_URL>/systemone`
- `eval/laya/bangun_dataset.py`: `--pesan`/`--label` boleh diulang
- `eval/laya/set_uji.py` (`SET_R3`), `eval/laya/set_kalibrasi.py` (`SET_R3`)
- `eval/laya/latih_kaggle.ipynb`: latih, ukur (kalibrasi lalu uji), dan sel keputusan
- `eval/laya/latih_laya.py`, `eval/laya/evaluasi.py`, `eval/laya/sajikan_laya.py`
- `eval/laya/data/keputusan.jsonl`: 40 keputusan tinjauan (29 r1 + 11 r3)
- `eval/laya/data/r3/`: dataset, `kaggle.zip`, dan hasil baseline r1
- `app/rag/gate.py`: `INSTRUCTIONS`, `CRITERIA`, `build_request`, `parse_response`
