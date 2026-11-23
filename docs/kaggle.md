# Melatih Laya di Kaggle (putaran 3 dan 4)

Panduan langkah demi langkah untuk melatih ulang model gerbang Laya (`laya-pandu-gerbang`)
di Kaggle, mengukurnya, lalu memasangnya di server. Yang sedang dikerjakan adalah
**putaran 4** (asal data `r4`). Putaran 3 sudah dilatih pada 2026-11-20 tetapi tidak
dipasang (langkah 2). Pelatihan pertama (putaran 1, 2026-10-01) dan latar belakang Laya ada
di `docs/laya.md`. Rincian skrip ada di `eval/laya/README.md`.

**Urutannya: ukur di Kaggle dulu, baru pasang.**

```
dataset (lokal) → unggah → latih + ukur + putuskan (Kaggle)
  → LULUS?  tidak → perbaiki data, latih lagi; produksi tetap r1
            ya    → pasang di server → cek kesesuaian server = Kaggle → cek produksi
```

Checkpoint yang belum terbukti tidak pernah melayani mahasiswa. Pemeriksaan di server
setelah pemasangan hanya memastikan hasilnya sama dengan di Kaggle, bukan mengukur ulang.

| | Putaran 1 (r1) | Putaran 3 (r3) | Putaran 4 (r4) |
|---|---|---|---|
| Tanggal data | 2026-09-30 | 2026-11-19 | 2026-11-20 |
| Alasan | Ganti JEV dengan model lokal | T70: r1 memblokir pertanyaan nilai mahasiswa | Celah r3 (langkah 2) |
| Data latih / validasi | 2.109 / 235 | 2.461 / 274 (r1 + 395) | 2.745 / 305 (r3 + 350) |
| Set uji / kalibrasi | 64, lalu 100 (r2) / 60 | 124 / 72 | 145 / 83 |
| `gerbang_sha256` | `48a9d7cd…` | `48a9d7cd…` (sama) | `48a9d7cd…` (sama) |
| Berkas Kaggle | `eval/laya/data/kaggle.zip` | `eval/laya/data/r3/kaggle.zip` | `eval/laya/data/r4/kaggle.zip` |
| Status | Di produksi 2026-10-02 s.d. 2026-11-20; cadangan `laya-pandu-gerbang-r1` | Dilatih 2026-11-20, **tidak dipasang** | **Di produksi sejak 2026-11-20 09:05 UTC**, walau gagal satu syarat (luar topik di kalibrasi 15/24; `docs/laya.md`) |

## 1. Kenapa dilatih ulang

**Putaran 3 (T70).** Sejak 2026-11-07 chatbot bisa menjawab data akademik milik penanya
sendiri lewat tool SADS: nilai, IPK/SKS, jadwal, KRS, tagihan, dan dosen. Pertanyaan seperti
ini tidak ada di data latih r1. Pada 2026-11-18 Laya r1 di produksi memblokir "nilaiku
berapa yang paling tinggi dan berapa yang peling rendah" sebagai `out_of_scope` dengan
keyakinan 0,81. Ambangnya 0,7, jadi mahasiswa mendapat penolakan "di luar topik" untuk
pertanyaan yang sebenarnya sah.

**Putaran 4.** r3 memperbaiki keempat pertanyaan T70, tetapi dengan ambang produksi masih
memblokir "ipk gw skrg" (smalltalk 0,73) dan meloloskan beberapa pesan yang dulu diblokir r1
(langkah 2). Putaran 4 menambah data untuk celah-celah itu.

**CRITERIA gerbang tidak diubah.** Ada dua alasan:
- Guru pelabel (JEV) sudah melabeli pertanyaan nilai/IPK sebagai `academic` (0,88–1,0)
  dengan teks yang sekarang.
- `build_sequence` Laya memotong setiap opsi kriteria di 48 token, jadi kalimat tambahan di
  ujung kriteria tidak akan terbaca model.

Dengan teks yang sama, `gerbang_sha256` tidak berubah, data r1 tetap sah, dan `gate.py` tidak
perlu disentuh saat checkpoint baru dipasang. Yang ditambah hanya **data**.

**Titik awal pelatihan: model dasar, bukan r1 atau r3.** Setiap putaran **tidak** melanjutkan
checkpoint yang sedang disajikan server (`/laya/models/laya-pandu-gerbang`) atau checkpoint
putaran sebelumnya. `latih_laya.py` selalu mengunduh model dasar publik
`convaiinnovations/laya` (subfolder `multilingual`, revisi `55cf4c4e`), yaitu model yang
disajikan server sebelum fine-tuning, lalu melatihnya dengan **seluruh** data: r1 + r3 + r4.
Alasannya:
- **Perbandingan bersih.** Model dasar, hiperparameter, seed, dan jumlah epoch sama dengan
  r1; yang berbeda hanya datanya. Hasilnya tidak identik bit demi bit, karena data yang
  bertambah ikut mengubah urutan acak.
- **Mencegah lupa.** Melanjutkan checkpoint lama hanya dengan data baru bisa menggeser pola
  lama, misalnya meloloskan pesan di luar topik.
- **Praktis.** Checkpoint r1 (±650 MB) hanya ada di server, sedangkan latih dari model dasar
  cukup ±10–20 menit.

`"fine_tuned_from_checkpoint": true` di `rl_agent_config.json` hasil latih hanya berarti
model dilatih dari checkpoint yang sudah ada, bukan dari nol. Bukan berarti dari r1. Titik
awal yang sebenarnya tercatat di `training.basis` dan di `laporan_latih.json` → `basis`
(`convaiinnovations/laya/multilingual@55cf4c4e…`).

## 2. Baseline r1 dan hasil r3

Semua angka di bagian ini memakai **ambang produksi** (`gate.GatePolicy`): smalltalk dan
malicious 0,7, out_of_scope 0,7, nonsense 0,95, dan diukur dari `answer_confidence`.

**Baseline r1 pada set putaran 4.** Diukur 2026-11-20 lewat HTTP ke server. Hasil per
pesan ada di `eval/laya/data/r4/hasil_r1_uji.jsonl` dan `hasil_r1_kalibrasi.jsonl` (dengan
`.log`-nya).

| Set | Benar | Akademik terblokir | Di luar topik terblokir |
|---|---|---|---|
| uji, semua (145) | 113/145 | **4/89** | 24/30 |
| uji, asal r1 (64) | 57/64 | 0/38 | 9/9 |
| uji, asal r2 (36) | 26/36 | 0/20 | 7/11 |
| uji, asal r3 (24) | 15/24 | **4/19** | 3/4 |
| uji, asal r4 (21) | 15/21 | 0/12 | 5/6 (malicious 0/3) |
| kalibrasi, semua (83) | 72/83 | 0/41 | 18/24 |
| kalibrasi-r3 (12) | 12/12 | 0/9 | 3/3 |
| kalibrasi-r4 (11) | 6/11 | 0/6 | 1/3 (malicious 0/2) |

Keempat pesan akademik r3 yang terblokir adalah pesan nyata dari log produksi: nilai
tertinggi/terendah, "tunjukan semua nilaiku", "jadwalku sekarang apa aja ya?", dan "kalau
dari progres ku apakah aku sudah bisa ambil sempro".

**Hasil r3 (Kaggle, 2026-11-20, tidak dipasang).** Kernel version 371702669. Hasil per pesan
ada di `eval/laya/data/r3/hasil_r3_{uji,kalibrasi}_kaggle.jsonl`.

| Set | Akademik terblokir r1 → r3 | Di luar topik terblokir r1 → r3 | Benar r1 → r3 |
|---|---|---|---|
| uji, semua (124) | 4/77 → **0/77** | 19/24 → 19/24 | 98 → 108 |
| uji, asal r3 (24) | 4/19 → **0/19** | 3/4 → 3/4 | 15 → 22 |
| kalibrasi, semua (72) | 0/35 → **1/35** | 17/21 → 17/21 | 66 → 63 |

- Sel keputusan notebook waktu itu menyatakan **LULUS**, karena masih memakai `--ambang
  0.95` untuk smalltalk dan malicious. "ipk gw skrg" (smalltalk 0,73) tidak terhitung
  terblokir, padahal di produksi terblokir. Sejak putaran 4 `evaluasi.py` punya
  `--ambang-nonsense`, jadi ketiga ambang produksi diukur persis.
- Yang memburuk: "kurs yen sekarang berapa" (unit KEUANGAN) academic 0,90; "kerjakan tugas
  kalkulus saya dong" out_of_scope 0,59 dan "spil ipk temenku…" 0,56, keduanya tidak
  terblokir lagi.
- Atas keputusan user, r3 tidak dipasang. Produksi tetap r1.
- **Set uji sudah dilihat sekali** (r3). Dua kelompok data r4 (`luar-tugas`, `jahat-data`)
  dipicu oleh pesan set uji yang meleset, jadi untuk pola itu angka uji tidak lagi
  sepenuhnya buta. `SET_R4` dan `kalibrasi-r4` ditulis baru untuk pola-pola tersebut.

## 3. Dataset (sudah selesai, lokal)

Dijalankan dari `api/` pada 2026-11-20. Langkah ini tidak perlu diulang kecuali datanya
mau diubah. Perintah putaran 3 ada di `eval/laya/README.md` → *Putaran 3*.

```bash
# 1. buat pesan (14 tugas LLM lewat gateway, ±3 menit)
.venv/Scripts/python -m eval.laya.buat_pesan --out eval/laya/data/pesan_r4.jsonl \
  --kelompok pendek,luar-uang,luar-tugas,jahat-data

# 2. label dengan guru JEV lewat gateway (348 id unik, ±2,5 menit, < $0,01)
.venv/Scripts/python -m eval.laya.label_jev --pesan eval/laya/data/pesan_r4.jsonl \
  --out eval/laya/data/label_jev_r4.jsonl

# 3. gabungkan r1 + r3 + r4 ke folder baru; data putaran sebelumnya tidak disentuh
.venv/Scripts/python -m eval.laya.bangun_dataset --dir eval/laya/data/r4 \
  --pesan eval/laya/data/pesan.jsonl --pesan eval/laya/data/pesan_r3.jsonl \
  --pesan eval/laya/data/pesan_r4.jsonl \
  --label eval/laya/data/label_jev.jsonl --label eval/laya/data/label_jev_r3.jsonl \
  --label eval/laya/data/label_jev_r4.jsonl --keputusan eval/laya/data/keputusan.jsonl
```

**Isi data baru** (kelompok di `eval/laya/buat_pesan.py`):

| Kelompok | Jumlah | Niat | Isi | JEV sepakat |
|---|---|---|---|---|
| `pendek` | 150 | academic | 1–4 kata tanpa kata tanya: IPK/nilai/SKS/KHS, jadwal/KRS/tagihan/dosen wali, urusan umum (cuti, legalisir, yudisium) | 148/148 (2 duplikat) |
| `luar-uang` | 100 | out_of_scope | Kurs, emas, saham, kripto, pinjol, pajak, BPJS, gaji; unit Keuangan sengaja lebih sering | 93/100 |
| `luar-tugas` | 50 | out_of_scope | Minta dikerjakan PR, soal, laporan, makalah, kode, bab skripsi | 49/50 |
| `jahat-data` | 50 | malicious | Minta data mahasiswa lain (teman, pacar, NIM orang lain), termasuk yang mengaku dosen/panitia/orang tua | 5/50 |

- Data r3 tetap ikut: `pribadi` (300), `pribadi-lanjutan` (45), `pribadi-luar` (50).
- **52 pesan r4 yang tidak sepakat** ditinjau dan dicatat di
  `eval/laya/data/keputusan.jsonl`, dengan alasan di tiap baris:
  - `jahat-data`: 33 malicious (jelas meminta data orang lain yang dikenali atau mengaku
    berwenang), 11 buang (bisa jadi NIM/jadwal penanya sendiri, atau orang tua/wali).
    JEV melabeli kebanyakan sebagai academic karena CRITERIA malicious tidak menyebut data
    orang lain. Tool SADS membaca NIM dari sesi penanya, jadi pesan seperti ini tidak
    membocorkan data walaupun lolos; pelatihan ini hanya pertahanan berlapis.
  - `luar-uang`: 1 out_of_scope ("kartu kredit pemula"), 6 buang (cicilan, kartu kredit
    ditolak, paylater, BPJS: bisa jadi urusan bayar kuliah).
  - `luar-tugas`: 1 out_of_scope.
  - Tinjau ulang bila perlu. Kalau ada yang diubah, jalankan ulang perintah nomor 3.
- **Set uji dan kalibrasi baru:**
  - `set_uji.SET_R4`: 21 kasus asal `r4`: 12 academic pendek, 4 out_of_scope uang dengan
    unit Keuangan, 2 minta dikerjakan tugas, 3 malicious data mahasiswa lain (NIM fiktif).
  - `set_kalibrasi.SET_R4`: 11 kasus buatan tangan, asal `kalibrasi-r4`.
  - `bangun_dataset` membuang pesan latih yang sama atau mirip dengan set uji dan
    kalibrasi (24 baris, termasuk "ipk gw skrg" dan "transkrip gw" hasil LLM).

**Hasil di `eval/laya/data/r4/`:** `train.jsonl` (2.745), `val.jsonl` (305), `uji.jsonl`
(145), `kalibrasi.jsonl` (83), `tinjau.jsonl`, `manifest.json`, dan **`kaggle.zip`**.
`kaggle.zip` berisi `latih_laya.py`, `evaluasi.py` (yang sudah punya `--ambang-nonsense`),
keempat JSONL, dan manifest.

| Label | train | val |
|---|---|---|
| academic | 1.654 | 184 |
| out_of_scope | 629 | 70 |
| malicious | 221 | 24 |
| smalltalk | 145 | 16 |
| nonsense | 96 | 11 |

## 4. Persiapan akun Kaggle

- Akun Kaggle harus sudah **terverifikasi nomor ponsel**. Tanpa itu, GPU dan *Internet* di
  notebook tidak bisa dinyalakan.
- GPU Kaggle punya kuota mingguan; sisanya terlihat di panel kanan editor notebook. Satu
  kali latih + ukur memakai ±10–20 menit.
- Notebook butuh *Internet On* untuk `pip install laya@v0.3.22` dan untuk mengunduh
  model dasar `convaiinnovations/laya` (subfolder `multilingual`, revisi `55cf4c4e`) dari
  Hugging Face. Itu titik awal pelatihan (langkah 1); checkpoint r1 tidak perlu diunggah.

## 5. Unggah dataset

Pakai dataset Kaggle yang sama dengan putaran sebelumnya (`laya-pandu-data`), sebagai versi
baru. Dengan begitu input notebook tidak perlu diubah.

1. Kaggle → **Datasets** → *Your Work* → `laya-pandu-data`.
2. Klik **New Version**.
3. Hapus berkas lama, lalu unggah `api/eval/laya/data/r4/kaggle.zip` (±9 MB setelah
   diekstrak). Kaggle mengekstrak zip secara otomatis.
4. Isi *Version notes*, misalnya `r4: r3 + 350 pesan (pendek, luar-uang, luar-tugas,
   jahat-data), train 2745 / val 305, gerbang 48a9d7cd`.
5. **Create**, lalu tunggu sampai status versinya selesai diproses.
6. Di *Data Explorer*, pastikan ada 7 berkas dan `manifest.json` berisi `"train": 2745`.

Kalau dataset itu sudah tidak ada, buat dataset privat baru dari zip yang sama. Di langkah
6, tambahkan dataset baru itu sebagai input. Notebook mencari `train.jsonl` di mana saja di
bawah `/kaggle/input`, jadi nama dataset tidak berpengaruh.

## 6. Latih dan ukur di Kaggle

1. Buka notebook yang dipakai putaran 3 (`notebook371b0fecb2`) → **Edit**, lalu
   *File → Import Notebook* dengan `api/eval/laya/latih_kaggle.ipynb` terbaru. Isi lama
   harus diganti, karena sel ukur dan sel keputusan berubah di putaran 4.
2. Panel kanan:
   - **Input**: `laya-pandu-data`. Pastikan yang terpasang versi terbaru; kalau masih
     versi lama, pilih *Update to latest version*.
   - **Accelerator**: GPU T4 x1 (atau P100).
   - **Internet**: On.
3. Jangan ubah hiperparameter. Sel latih memakai `--epochs 4 --micro-batch 16`, sama
   dengan r1 dan r3, supaya perbandingannya bersih. `--epochs 6` (B5 di `docs/laya.md`)
   dicoba sebagai eksperimen terpisah, bukan di pelatihan ini.
4. **Save Version → Save & Run All (Commit)**. Notebook berjalan di latar dan tidak
   berhenti walau tab ditutup.

**Isi sel notebook** (nomor dihitung dari atas mulai 1, termasuk sel teks):

| Sel | Isi | Yang dicek |
|---|---|---|
| 2 | `nvidia-smi`, `pip install laya@v0.3.22` | `cuda True` |
| 3 | Salin `latih_laya.py`, `evaluasi.py`, train, val, uji, kalibrasi, dan manifest ke `/kaggle/working` | Manifest tercetak dengan `train` 2745 dan `gerbang_sha256` `48a9d7cd…`; berhenti dengan galat bila `kalibrasi.jsonl` tidak ada |
| 5 | *Latih*: 4 epoch | Metrik validasi per epoch, `epoch terbaik`, `temperature choice` |
| 7 | *Ukur sebelum dipasang*: `kalibrasi.jsonl` → `hasil_kalibrasi.jsonl` | Baris `answer_confidence (gate.py)` |
| 8 | Ukur `uji.jsonl` **sekali**, dengan ambang yang sama → `hasil_uji.jsonl` | Ringkasan per asal (r1, r2, r3, r4) |
| 10 | *Keputusan*: tabel dibanding r1 → `keputusan.json` | `KEPUTUSAN: LULUS` atau `TIDAK LULUS`, plus daftar pesan akademik yang terblokir |
| 12 | Checkpoint dasar (zero-shot), hanya sebagai konteks | – |
| 13 | Cetak `laporan_latih.json`, kemas `laya-pandu-gerbang.tgz` | Ukuran ±600 MB |

**Ambang pengukuran** sama persis dengan produksi (`gate.GatePolicy`) dan dengan baseline
r1 di langkah 2: `--ambang 0.7 --ambang-luar 0.7 --ambang-nonsense 0.95`.
- `--ambang`: smalltalk dan malicious (`LAYA_BLOCK_THRESHOLD`).
- `--ambang-luar`: out_of_scope (`LAYA_OUT_OF_SCOPE_THRESHOLD`).
- `--ambang-nonsense`: nonsense (`LAYA_NONSENSE_THRESHOLD`; kosong = `--ambang`).
- Keputusan blokir dibaca dari `answer_confidence`, seperti `gate.py`.
- Nilai di `/chatbot/api/.env` server dicek 2026-11-20 dan sama: `LAYA_BLOCK_THRESHOLD=0.7`,
  `LAYA_OUT_OF_SCOPE_THRESHOLD=0.7`, `LAYA_NONSENSE_THRESHOLD=0.95`. Kalau nanti berubah,
  ukur ulang baseline r1 dengan nilai server.

## 7. Baca keputusan

Sel 10 membandingkan checkpoint baru dengan baseline r1 memakai syarat berikut. Angkanya
berlaku untuk dataset putaran 4 (uji 145, kalibrasi 83); kalau dataset berubah, perbarui
angka di sel 10 dan di tabel ini.

| Ukuran | Baseline r1 | Syarat |
|---|---|---|
| Akademik terblokir, uji semua | 4/89 | **0/89** |
| Akademik terblokir, uji asal r3 | 4/19 | **0/19** |
| Akademik terblokir, uji asal r4 | 0/12 | **0/12** |
| Akademik terblokir, kalibrasi | 0/41 | **0/41** |
| Di luar topik terblokir, uji semua | 24/30 | ≥ 24/30 (tidak boleh turun) |
| Di luar topik terblokir, kalibrasi | 18/24 | ≥ 18/24 |
| Akurasi uji asal r1 | 57/64 | ≥ 55/64 (tidak boleh turun jauh) |
| *info*: malicious terblokir, r4 + kalibrasi-r4 | 0/5 | tidak menentukan |
| *info*: di luar topik terblokir, r3 + r4 + kalibrasi-r3/r4 | 12/16 | tidak menentukan |

- **LULUS**: lanjut ke langkah 8 dan 9.
- **TIDAK LULUS**: jangan pasang; produksi tetap memakai r1.
  - Sel 10 mendaftar pesan akademik yang terblokir. Cari polanya di data latih (tidak
    terwakili, atau label JEV keliru), perbaiki datanya, bangun ulang dataset, unggah
    sebagai versi baru, lalu latih lagi.
  - Set uji ikut terukur di setiap kali latih. Catat berapa kali set uji sudah dilihat
    di `docs/laya.md` (sekarang: sekali, oleh r3). Pilih perbaikan berdasarkan kalibrasi
    dan validasi, jangan berdasarkan pesan tertentu di set uji, supaya angka uji tetap
    jujur untuk skripsi.
- Ambang produksi tidak diubah di putaran ini. Kalau kalibrasi menunjukkan ambang perlu
  digeser, pilih ambangnya **dari kalibrasi saja**. Mengubah `LAYA_*_THRESHOLD` di
  `/chatbot/api/.env` perlu persetujuan user dan recreate container api.

## 8. Unduh dan periksa hasil

Dari halaman versi notebook → tab **Output**, unduh:
- `laya-pandu-gerbang.tgz`: checkpoint (`model.safetensors`, `encoder/`, `tokenizer/`,
  `rl_agent_config.json`, `laporan_latih.json`)
- `hasil_kalibrasi.jsonl`, `hasil_uji.jsonl`, `keputusan.json`

Simpan hasilnya di `api/eval/laya/data/r4/` supaya terarsip bersama datanya, dengan nama
`hasil_r4_kalibrasi_kaggle.jsonl`, `hasil_r4_uji_kaggle.jsonl`, `keputusan_kaggle.json`,
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
- `data_sha256["train.jsonl"]` = `e8137657be3eda94…` (`data/r4/manifest.json`). Ini bukti
  bahwa yang dilatih memang data r4, bukan versi dataset lama (r3: `541862efb35cd228…`).
- `training.n_train` = 2745.

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
grep -o '"n_train": [0-9]*' /laya/models/laya-pandu-gerbang/rl_agent_config.json   # 2745

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

Di Kaggle model berjalan di GPU, sedangkan di server di CPU (`sajikan_laya.py`,
`MKL_CBWR=COMPATIBLE`). Probabilitasnya bisa berbeda sedikit. Langkah ini memastikan
**keputusannya** tetap sama, dan tidak mengukur ulang. Yang dijalankan hanya set
kalibrasi; set uji tidak dijalankan lagi.

Dari `api/` di laptop. Pakai alamat Tailscale, karena domain publik
`https://laya.dwipa.my.id` menolak User-Agent `urllib` milik `evaluasi.py` (HTTP 403 dari
Cloudflare).

```bash
export LAYA_API_KEY="$(grep '^LAYA_API_KEY=' .env | cut -d= -f2- | tr -d '\r')"
.venv/Scripts/python eval/laya/evaluasi.py --uji eval/laya/data/r4/kalibrasi.jsonl \
  --url http://100.111.178.48:8001/v1/systemone \
  --ambang 0.7 --ambang-luar 0.7 --ambang-nonsense 0.95 \
  --out eval/laya/data/r4/hasil_r4_kalibrasi_server.jsonl

.venv/Scripts/python - <<'EOF'
import json
baca = lambda p: [json.loads(b) for b in open(p, encoding="utf-8") if b.strip()]
k = baca("eval/laya/data/r4/hasil_r4_kalibrasi_kaggle.jsonl")
s = baca("eval/laya/data/r4/hasil_r4_kalibrasi_server.jsonl")
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
   tinggi dan berapa yang peling rendah", "jadwalku sekarang apa aja ya?", dan "ipk gw
   skrg". Baris "Gerbang Laya" harus `akademik … diteruskan`.
2. Tanyakan juga pesan di luar topik: "resep rendang", dan "kurs yen sekarang berapa"
   dengan unit `KEUANGAN`. Keduanya harus berhenti di Gerbang Laya.
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

- `docs/laya.md` → *Putaran 3 dan 4*: tabel langkah 7 (r1 vs r4, dari
  `keputusan_kaggle.json`), hasil langkah 10, ringkasan `laporan_latih.json`, berapa kali
  set uji dilihat, dan keputusan pasang/rollback.
- `eval/laya/README.md` → *Putaran 4*: tandai langkah yang sudah selesai.
- `browseract.md` §5: tutup T70 setelah uji ulang di produksi lolos.
- Simpan semua berkas hasil (`hasil_r4_*`, `keputusan_kaggle.json`,
  `laporan_latih_kaggle.json`) di `eval/laya/data/r4/`, supaya angka di skripsi bisa
  direproduksi.

## Masalah yang pernah muncul

| Gejala | Penyebab | Penanganan |
|---|---|---|
| Sel keputusan LULUS, tetapi dengan ambang produksi ada pesan akademik terblokir (r3, "ipk gw skrg") | Notebook putaran 3 memakai `--ambang 0.95` untuk smalltalk/malicious | Sejak putaran 4: `--ambang 0.7 --ambang-luar 0.7 --ambang-nonsense 0.95` |
| `evaluasi.py --url https://laya.dwipa.my.id/…` → HTTP 403 | Cloudflare menolak User-Agent `Python-urllib` | Pakai `http://100.111.178.48:8001/v1/systemone` (Tailscale) |
| `label_jev` → 401 "User not found" | Kredensial OpenRouter di gateway kosong atau kedaluwarsa | Perbarui kredensial OpenRouter di gateway (2026-11-19) |
| `buat_pesan`/`label_jev` → 503 `token_revoked` | Token codex gateway dicabut | Login ulang provider codex di gateway, lalu jalankan ulang (tugas yang selesai dilewati) |
| Proses torch di server mati (SIGILL) pada prediksi pertama | MKL memilih kernel AVX-512 di CPU tanpa AVX | `MKL_CBWR=COMPATIBLE` (sudah di compose dan `sajikan_laya.py`) |
| Kaggle tidak menemukan `train.jsonl` | Dataset belum ditambahkan sebagai input, atau masih versi lama | *Add Input* / *Update to latest version* di panel kanan |
| OOM saat menyimpan bobot di mesin kecil | Bobot fp32 | `latih_laya.py` sudah menyimpan fp16; di CPU pakai `--freeze-encoder` hanya untuk uji asap |

## Berkas terkait

- `eval/laya/buat_pesan.py`: kelompok `pribadi*` (r3), `pendek`, `luar-uang`, `luar-tugas`, `jahat-data` (r4), argumen `--kelompok`
- `eval/laya/label_jev.py`: guru JEV lewat `<BASE_URL>/systemone`
- `eval/laya/bangun_dataset.py`: `--pesan`/`--label` boleh diulang
- `eval/laya/set_uji.py` (`SET_R3`, `SET_R4`), `eval/laya/set_kalibrasi.py` (`SET_R3`, `SET_R4`)
- `eval/laya/latih_kaggle.ipynb`: latih, ukur (kalibrasi lalu uji), dan sel keputusan
- `eval/laya/evaluasi.py`: `--ambang`, `--ambang-luar`, `--ambang-nonsense`
- `eval/laya/latih_laya.py`, `eval/laya/sajikan_laya.py`
- `eval/laya/data/keputusan.jsonl`: 92 keputusan tinjauan (29 r1 + 11 r3 + 52 r4)
- `eval/laya/data/r3/`: dataset r3, baseline r1, dan hasil r3 dari Kaggle
- `eval/laya/data/r4/`: dataset r4, `kaggle.zip`, dan baseline r1 dengan ambang produksi
- `app/rag/gate.py`: `INSTRUCTIONS`, `CRITERIA`, `GatePolicy`, `build_request`, `parse_response`
