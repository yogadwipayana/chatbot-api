# Backend — Chatbot Administrasi Mahasiswa

Implementasi PRD v2.0 bagian backend. FastAPI + LangChain (LCEL) di atas
PostgreSQL 16 + pgvector.

## Menjalankan

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
cp .env.example .env          # lalu isi kunci API dan ADMIN_JWT_SECRET
alembic upgrade head
uvicorn app.main:app --reload
```

## Test

```bash
pytest                        # unit + API; test integrasi dilewati otomatis
pytest tests/unit -q          # cepat, tanpa dependensi luar
pytest --cov=app --cov-report=term-missing

# Test integrasi butuh Postgres hidup:
TEST_DATABASE_URL=postgresql+asyncpg://chatbot:chatbot@localhost:5432/chatbot_test \
  pytest tests/integration
```

Tidak ada satu pun test yang memanggil API LLM, API embedding, atau LangSmith.
Selain soal biaya, test yang menyentuh jaringan tidak bisa membuktikan invarian
"LLM tidak dipanggil" — yang justru inti FR-3 dan FR-7.

## Kredensial model AI

Chat dan embedding memakai satu endpoint OpenAI-compatible:

| Variabel | Isi |
|---|---|
| `BASE_URL` | Endpoint API. Kosong = OpenAI resmi; penyedia lain umumnya diakhiri `/v1`. |
| `API_KEY` | Kunci endpoint tersebut. Wajib saat `ENVIRONMENT=production`. |
| `CHAT_MODEL` | ID model penjawab. |
| `EMBED_MODEL` | ID model embedding. Harus menghasilkan **1024 dimensi**. |

- Mengganti `EMBED_MODEL` setelah ada dokumen berarti re-index seluruhnya.
- Saat `BASE_URL` diisi, embedding dikirim sebagai teks mentah
  (`check_embedding_ctx_length=False`); default LangChain mengirim token ID
  yang ditolak banyak endpoint non-OpenAI.
- Model chat baru perlu tarifnya di `app/observability/costs.py` agar estimasi
  biaya (AD-5) berjalan.

## Tracing LangSmith (FR-8)

Menyala bila `LANGSMITH_TRACING=true` **dan** `LANGSMITH_API_KEY` terisi. Status
sebenarnya dilaporkan `GET /health` (`tracing_enabled`) dan satu baris log saat
proses start -- tracing yang mati tidak menjatuhkan permintaan apa pun, jadi
tanpa dua tanda itu ia baru ketahuan saat ada jawaban buruk yang jejaknya
ternyata tidak pernah dikirim.

- **Satu giliran tanya-jawab = satu trace.** Penulisan ulang query (FR-4),
  retrieval (FR-2), dan penyusunan jawaban (FR-5) menjadi child run di bawah
  satu akar. `messages.langsmith_run_id` menyimpan ID akar itu, bukan ID
  panggilan LLM di dalamnya.
- **Kolom itu `NULL` saat tracing mati**, dan memang harus begitu: ID yang
  dicatat tanpa trace yang terkirim hanya menghasilkan tautan buntu di AD-4.
- **`session_id` ikut di setiap run**, bukan hanya di akar, supaya trace satu
  percakapan dapat dikelompokkan sebagai thread dan biayanya dijumlahkan.
- **`LANGSMITH_ENDPOINT`** hanya perlu diisi untuk region Eropa atau instans
  self-hosted.
- Variabel `LANGCHAIN_TRACING` / `LANGCHAIN_TRACING_V2` yang tertinggal di
  environment sengaja **dihapus** saat start: langsmith membacanya lebih dulu
  daripada `LANGSMITH_TRACING`, sehingga sisa variabel dari proyek lain bisa
  menyalakan tracing yang sudah dimatikan, atau sebaliknya.

## Penyimpanan dokumen PDF

Dua backend, dipilih lewat `STORAGE_BACKEND`:

- **`local`** (default) — disk server. Tanpa kredensial, dipakai pengembangan
  dan test. Jangan untuk produksi: berkas ikut hilang saat kontainer diganti,
  dan tidak bisa dibagi antar instans.
- **`s3`** — apa pun yang bicara protokol S3: **AWS S3**, **Cloudflare R2**,
  MinIO, Backblaze B2. Dibedakan hanya oleh `S3_ENDPOINT_URL` dan kredensial.

Kolom `documents.file_path` menyimpan **kunci objek** (`documents/<uuid>.pdf`),
bukan lintasan disk — pindah dari lokal ke R2 tidak memaksa migrasi data.
Kolom `documents.nama_file` menyimpan nama asli unggahan untuk nama tab browser
dan nama berkas saat PDF dilihat atau diunduh. Key objek tetap berbasis UUID agar
nama file tidak dapat menimpa unggahan lain.

### Kredensial yang diperlukan — Cloudflare R2

Empat nilai. Semuanya dari Cloudflare Dashboard:

| Variabel | Isi | Cara mendapatkan |
|---|---|---|
| `S3_ENDPOINT_URL` | `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` | **Account ID** ada di sidebar halaman R2 |
| `S3_BUCKET` | Nama bucket | **R2 → Create bucket** |
| `S3_ACCESS_KEY_ID` | Access Key ID | **R2 → Manage R2 API Tokens → Create API Token** |
| `S3_SECRET_ACCESS_KEY` | Secret Access Key | Sama; **hanya ditampilkan sekali** |

Saat membuat token: permission **Object Read & Write**, dan batasi scope-nya ke
bucket ini saja — bukan seluruh akun.

> R2 API token ≠ Cloudflare API token biasa. Yang dipakai di sini khusus dari
> menu **R2 → Manage R2 API Tokens**; token akun biasa tidak akan bekerja.

```bash
STORAGE_BACKEND=s3
S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
S3_REGION=auto
S3_BUCKET=dokumen-kampus
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
```

Verifikasi sebelum menyatakan selesai — script ini menulis, membaca, membuat
URL, lalu menghapus satu objek uji di bucket sungguhan:

```bash
python -m scripts.check_storage
```

### Tiga hal khas R2 yang sudah ditangani

**1. `S3_REGION` harus `auto`.** R2 tidak punya region, tetapi SDK menuntut
nilai. Nilai kosong dan `us-east-1` di-alias ke `auto`; selain itu ditolak
validator konfigurasi saat startup.

**2. Checksum.** Sejak botocore 1.36 default `request_checksum_calculation`
adalah `when_supported`, yang menempelkan CRC32 **FULL_OBJECT** pada setiap
unggahan. R2 hanya mendukung CRC32 sebagai **COMPOSITE**, sehingga unggahan
dapat ditolak. `S3_CHECKSUM_COMPAT=true` (default) menurunkannya ke
`when_required`. Biarkan menyala kecuali memakai AWS S3 asli.

**3. Endpoint R2 bukan untuk peramban.** `r2.cloudflarestorage.com` adalah
endpoint API. Untuk menyajikan PDF ke mahasiswa (FE-2) ada dua pilihan:

| | Kapan dipakai |
|---|---|
| **Presigned URL** (default) | Bucket privat. Tautan berumur `S3_PRESIGN_TTL_SECONDS` (default 15 menit), berubah tiap kali, tidak dapat di-cache. |
| **`S3_PUBLIC_BASE_URL`** | Bucket dipublikasikan lewat domain kustom atau URL r2.dev. Tautan stabil dan dapat di-cache CDN. |

### AWS S3 dan MinIO

AWS S3: kosongkan `S3_ENDPOINT_URL`, isi `S3_REGION` dengan region sungguhan
(mis. `ap-southeast-1`). Kredensialnya IAM access key dengan izin
`s3:GetObject`, `s3:PutObject`, `s3:DeleteObject` pada bucket tersebut.

MinIO: `S3_ENDPOINT_URL=http://localhost:9000` dan
`S3_ADDRESSING_STYLE=path` — MinIO umumnya tidak melayani virtual-host style.

Contoh lengkap ketiganya ada di `.env.example`.

## Kontrak API — `api.yaml`

OpenAPI 3.1.0, mencakup **seluruh** permukaan API PRD termasuk endpoint yang
belum ditulis, supaya pekerjaan `client/` dan `admin/` tidak perlu menunggu
backend selesai. Setiap operasi diberi `x-status`:

- `implemented` — sudah ada di `app/routers/`, diuji di `tests/api/`
- `planned` — kontrak disepakati, kode belum ditulis

`tests/api/test_openapi_contract.py` menjaganya tetap jujur: memvalidasi berkas
terhadap meta-schema OpenAPI 3.1 resmi, lalu membandingkan operasi
`implemented` dengan spesifikasi yang benar-benar dihasilkan FastAPI — dua arah.
Endpoint baru yang lupa didokumentasikan, atau kontrak yang menyimpang dari
model Pydantic, menggagalkan test.

Kalau kode dan `api.yaml` berbeda untuk endpoint `implemented`, **kode yang
benar** — perbarui YAML-nya.

```bash
# Lihat spesifikasi yang dihasilkan aplikasi, untuk membandingkan
python -m scripts.export_openapi
python -m scripts.export_openapi -o /tmp/live.yaml

# Swagger UI interaktif ada di /docs saat server jalan (nonaktif di produksi)
```

## Peta kode

| Berkas | Isi | PRD |
|---|---|---|
| `app/rag/fusion.py` | Reciprocal Rank Fusion | FR-2 |
| `app/rag/retriever.py` | `PostgresHybridRetriever` — vector + FTS paralel | FR-2 |
| `app/rag/filters.py` | Predikat dokumen aktif, satu definisi untuk semua query | FR-2 |
| `app/rag/threshold.py` | Ambang penolakan, dievaluasi sebelum LLM | FR-3 |
| `app/rag/rewriter.py` | Penulisan ulang query | FR-4 |
| `app/rag/prompts.py` | Instruksi wajib + penyusunan konteks | FR-5 |
| `app/rag/risk.py` | Topik berisiko tinggi + kontak unit | FR-6 |
| `app/rag/sensitive.py` | Pertanyaan sensitif → konseling | FR-7 |
| `app/rag/chain.py` | Orkestrasi alur, termasuk jalan keluar lebih awal | §7 |
| `app/ingestion/` | Loader, chunker, embedder, pipeline | FR-1 |
| `app/storage/` | Penyimpanan objek: disk lokal / S3 / R2 | FR-1, FE-2 |
| `app/observability/` | LangSmith + estimasi biaya | FR-8 |
| `app/security/` | Sanitasi, rate limit, kill switch, auth | FR-9, AD-1 |
| `eval/` | Recall@k, MRR, kalibrasi ambang | §3, §14 |

Modul `fusion`, `threshold`, `risk`, `sensitive`, `citations`, `filters`, dan
`eval/metrics` sengaja **tanpa impor pihak ketiga**. Ini bagian yang PRD §6
sebut sebagai kontribusi teknis: ia harus dapat dijelaskan dan diuji tanpa
bergantung pada LangChain, dan tidak ikut rusak saat LangChain naik versi.

## Tiga keputusan yang perlu diketahui sebelum mengubah kode

**1. Ambang penolakan menilai skor mentah, bukan skor RRF.**
Skor RRF hanya mencerminkan peringkat. Chunk peringkat 1 selalu memperoleh skor
RRF maksimum, termasuk ketika seluruh kandidat tidak relevan — menilai ambang
atas skor RRF membuat sistem tidak pernah menolak apa pun, dan FR-3 mati diam-diam.
Karena itu `FusedHit` membawa `raw_scores` per sumber. Dijaga oleh
`tests/unit/test_threshold.py::TestSkorRRFTidakDipakai`.

**2. FR-7 berjalan sebelum retrieval, dan menang atas FR-6.**
Mahasiswa yang menulis "saya stres, takut di-DO" tidak boleh dibalas kutipan
pasal tata cara DO. Pemeriksaan sensitif adalah langkah pertama `run_pipeline`,
sebelum ada query ke database.

**3. Penolakan menampilkan semua unit terkait, bukan hanya yang pertama.**
"Deadline pembayaran UKT" menyangkut akademik dan keuangan sekaligus; mahasiswa
yang ditolak tidak boleh dikirim ke loket yang salah.

## Yang masih placeholder

Perlu diganti sebelum rilis — semuanya bergantung pada Fase 0 PRD §13:

- `app/rag/risk.py` → `DEFAULT_CONTACTS`: nama unit, jam layanan, kontak resmi
- `app/rag/sensitive.py` → `CRISIS_CONTACTS`, `DISTRESS_CONTACTS`: verifikasi nomor
- `ThresholdPolicy` default (0.35 / 0.05): ganti dengan hasil kalibrasi empiris
- `app/observability/costs.py` → `PRICES_PER_MTOK`: perbarui dari halaman harga resmi
- `FTS_CONFIG = "indonesian"`: pastikan tersedia di instans target
  (`SELECT cfgname FROM pg_ts_config`) — kalau tidak ada, seluruh jalur FTS
  gagal diam-diam. Test integrasi memeriksa ini.

## Setelan yang dapat diubah dari dashboard

Sembilan parameter retrieval dan chunking dapat disetel superadmin lewat menu
**Konfigurasi** di `../admin`, tanpa menyunting `.env` dan tanpa restart:

| Kelompok | Variabel |
|---|---|
| Pencarian | `RETRIEVAL_CANDIDATES`, `RETRIEVAL_TOP_N`, `RRF_K`, `RRF_WEIGHT_VECTOR`, `RRF_WEIGHT_FULLTEXT` |
| Ambang (FR-3) | `VECTOR_THRESHOLD`, `LEXICAL_THRESHOLD` |
| Chunking (FR-1) | `CHUNK_SIZE`, `CHUNK_OVERLAP` |

Cara kerjanya (`app/admin/runtime_config.py`, tabel `runtime_config`):

- **Hanya nilai yang ditimpa yang disimpan.** Parameter tanpa baris di tabel itu
  tetap mengikuti `.env`, jadi menyunting `.env` lalu restart masih berlaku untuk
  parameter yang belum pernah disentuh dari dashboard.
- **Nilainya disimpan sebagai teks** dan di-parse ulang oleh `Settings`, sehingga
  validasi -- termasuk `chunk_overlap < chunk_size` dan
  `retrieval_top_n <= retrieval_candidates` -- hanya ditulis satu kali.
- **Dibaca per permintaan** (`get_effective_settings` di `app/deps.py`), bukan
  di-cache di memori proses: perubahan berlaku seketika di semua worker.
  Endpoint yang tidak membutuhkannya memakai `BaseSettingsDep` supaya tidak
  membayar query tambahan.
- **Baris yang tidak dapat dipakai diabaikan, bukan menjatuhkan layanan.** Bila
  `.env` berubah sehingga kombinasinya melanggar aturan, layanan kembali memakai
  `.env` dan halaman Konfigurasi menampilkan alasannya.
- `CHUNK_SIZE` dan `CHUNK_OVERLAP` hanya mengenai dokumen yang diproses
  setelahnya; dokumen lama baru ikut berubah bila diunggah ulang.

Sisanya -- nama model, kredensial, CORS, batas unggah, zona waktu -- tetap hanya
lewat `.env` + restart. Mengganti `EMBED_MODEL` menuntut re-index seluruh
dokumen, jadi ia sengaja bukan setelan yang dapat diubah sambil layanan jalan.

```bash
# Melihat dan mengubah tanpa dashboard
curl -H "Authorization: Bearer $TOKEN" localhost:8000/api/admin/config
curl -X PATCH -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"vector_threshold": 0.42}' localhost:8000/api/admin/config
curl -X DELETE -H "Authorization: Bearer $TOKEN" localhost:8000/api/admin/config  # kembali ke .env
```

## Kalibrasi ambang (FR-3)

PRD menuntut ambang ditentukan empiris, bukan ditebak. Butuh dua kumpulan:
set evaluasi (pertanyaan yang memang terjawab) dan kumpulan negatif
(pertanyaan yang seharusnya ditolak — ambil dari tabel `unanswered` setelah
uji terbatas, jangan dikarang).

```bash
python -m eval.calibrate_threshold \
  --dataset eval/data/eval_set.jsonl \
  --negatives eval/data/negatives.txt
```

## Evaluasi retrieval (§14)

```bash
python -m eval.run_eval --dataset eval/data/eval_set.jsonl --k 5
```

Membandingkan baseline vector-only melawan hybrid + RRF lewat jalur SQL yang
sama persis (baseline = bobot fulltext nol), sehingga yang terukur benar-benar
hanya efek fusi. Exit code 1 bila Recall@5 belum mencapai 0.85.

## Dashboard admin

Endpoint AD-1..AD-6 (`app/routers/admin_*.py`) dipakai oleh `../admin`.

Sumber jawaban chatbot ada dua jenis, keduanya baris `documents` (lihat
`docs/schema.md`): dokumen PDF yang diunggah (`/api/admin/documents`) dan
entri tanya jawab yang diketik langsung (`/api/admin/faq`). Retrieval
memperlakukan keduanya sama; yang berbeda hanya cara admin menyuntingnya.

### CORS — saat front-end memakai domain lain

`admin/` dan `client/` boleh berada di domain yang berbeda dari API. Begitu itu
terjadi, `CORS_ORIGINS` di `.env` **wajib** menyebut setiap asal, dipisah koma
dan tanpa garis miring akhir:

```bash
CORS_ORIGINS=https://admin.dwipa.my.id,https://sads.dwipa.my.id
```

Yang perlu diingat:

- Nilai ini dibaca sekali saat proses start. Mengubahnya berarti **restart**
  (`pm2 restart api`); tanpa itu nilai lama masih dipakai.
- Asal harus cocok persis — skema, host, dan port. `https://sads.dwipa.my.id`
  tidak mencakup `http://`, subdomain lain, maupun port lain.
- Saat `ENVIRONMENT=local`, `http://localhost:3000` dan `:3001` selalu ikut
  diizinkan, jadi mengisi daftar produksi di `.env` pengembang tidak
  mematikan `npm run dev`.
- Daftar kosong di luar `local` berarti tidak ada pemanggil lintas-asal yang
  dilayani. Itu benar hanya bila API dan front-end berbagi domain di balik Caddy.

Gejala bila ini salah: peramban menolak dengan `No 'Access-Control-Allow-Origin'
header is present` dan permintaan tidak pernah sampai ke handler — log API bersih,
seolah-olah front-end tidak pernah memanggil.

### Level akses

| Level | Hak |
|---|---|
| `staf` (Staf/Dosen) | Kelola dokumen dan tanya jawab **unitnya sendiri**, uji coba jawaban, lihat pertanyaan tak terjawab |
| `admin` | + dokumen dan tanya jawab semua unit, tandai pertanyaan selesai, statistik, umpan balik mahasiswa |
| `superadmin` | + kill switch, konfigurasi retrieval dan chunking, kelola akun (`/api/admin/users`) |

Aturannya ada di `app/admin/permissions.py` (tanpa impor pihak ketiga). Setiap
operasi admin di `api.yaml` mencatat `x-min-role`, dan
`tests/api/test_admin_roles.py` membangkitkan satu kasus 403 untuk setiap level
di bawah minimum -- operasi baru yang lupa diberi pembatasan langsung gagal test.

- Level, unit, dan status aktif dibaca dari database per permintaan, bukan dari
  token: perubahan oleh superadmin berlaku seketika.
- Token yang terbit sebelum kata sandi terakhir diganti ditolak.
- Unit dicocokkan tanpa peduli huruf besar dan spasi berlebih. Staf melihat
  dokumen yang kolom `unit`-nya sama dengan unit akunnya; ejaan unit di dokumen
  dan di akun harus sepadan.
- Superadmin tidak dapat menurunkan level, menonaktifkan, atau menghapus dirinya
  sendiri, dan superadmin aktif terakhir tidak dapat disingkirkan.

Akun biasanya dibuat superadmin lewat menu **Admin** di dashboard, dengan kata
sandi sementara yang ditampilkan sekali. Superadmin pertama -- atau pemulihan
bila semua superadmin terkunci -- dibuat dari server:

```bash
python -m scripts.create_admin admin@kampus.ac.id --role superadmin   # kata sandi acak, tampil sekali
python -m scripts.create_admin keuangan@kampus.ac.id --role staf --unit "Biro Keuangan"
python -m scripts.create_admin admin@kampus.ac.id --reset             # sandi baru + aktifkan kembali
```

Data contoh untuk pengembangan: dokumen fiktif yang diproses lewat pipeline
ingestion sungguhan (memanggil API embedding dan menulis ke penyimpanan objek),
dan 30 hari log percakapan. Diberi penanda sehingga dapat dihapus tanpa
menyentuh data asli. Menolak berjalan saat `ENVIRONMENT=production`.

```bash
python -m scripts.seed_demo --dokumen --log
python -m scripts.seed_demo --hapus
```

Beberapa hal yang tidak terlihat dari kontrak:

- Setiap jawaban `/api/chat` dan `/api/chat/stream` dicatat ke
  `conversations`/`messages`, dan penolakan juga ke `unanswered` -- sumber data
  AD-4 dan AD-5. Pencatatan yang gagal tidak menggagalkan jawaban.
- Isi pertanyaan sensitif (FR-7) tidak disimpan; jumlahnya tetap tercatat.
- Uji coba admin (AD-6) tidak dicatat dan tidak tunduk pada kill switch.
- Pengelompokan AD-4 (`app/admin/grouping.py`) memakai kemiripan kata setelah
  kata tanya dibuang, bukan embedding, dan tanpa impor pihak ketiga.
- Estimasi biaya AD-5 hanya menghitung model yang tarifnya ada di
  `PRICES_PER_MTOK`. Jawaban dari model tanpa tarif dilaporkan sebagai
  `pesan_tanpa_estimasi_biaya`, bukan sebagai biaya nol.
- Kill switch dan pembatas login tersimpan di memori proses: benar untuk satu
  worker uvicorn (Dockerfile), harus dipindah ke Postgres bila worker ditambah.

## Belum diimplementasikan

Rate limiter untuk endpoint chat (login admin sudah dibatasi), saran pertanyaan
(FE-6, menunggu hasil survei), dan chain penulisan ulang query yang benar-benar
memanggil LLM (fungsi pembantunya sudah ada dan teruji).
