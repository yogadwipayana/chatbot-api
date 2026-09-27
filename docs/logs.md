# Rancangan: Log Aplikasi di SQLite + Halaman Log Admin

Status: **selesai** (API + halaman admin) · dicatat 2026-09-25

## Latar belakang

Kondisi logging `api/` saat ini:

| Lapis | Tempat | Keterangan |
|---|---|---|
| Log percakapan | Postgres (`conversations`, `messages`, `unanswered_questions`) | `app/observability/chatlog.py`; sumber dashboard AD-4/AD-5 |
| Tracing | LangSmith | `app/observability/tracing.py`; satu trace per giliran, node LangGraph jadi child run |
| Log Python (`logging`) | stdout | **tidak pernah dikonfigurasi** |

Masalah yang ingin diselesaikan:

1. Tidak ada `basicConfig`/`dictConfig`. Uvicorn hanya mengonfigurasi logger
   `uvicorn.*`, sehingga logger `app.*` jatuh ke *lastResort* Python: level
   INFO hilang (mis. `Tracing LangSmith aktif/mati` di `app/main.py`,
   `audit.info` di `admin_config.py`), dan WARNING/ERROR tercetak tanpa
   timestamp, level, atau nama logger.
2. Proses per node LangGraph hanya terlihat di LangSmith, dan hanya bila tracing
   menyala. Tidak ada catatan durasi/status per node di sisi aplikasi.
3. Log audit hanya ada di stdout, jadi hilang saat log pm2/Docker dirotasi.

Sudah dikerjakan sebelumnya: panggilan HTTP JEV kini tercatat sebagai child run
`jev_classify` di bawah node `jev_gate` (`app/rag/gate.py`, `@traceable`).

## Keputusan

- Penyimpanan: **SQLite** di `api/log/app.db` (masuk `.gitignore`).
- Akses halaman Log: **admin dan superadmin** (`min: "admin"`, sama seperti
  Statistik dan Biaya).
- Masa simpan: **7 hari**.
- Logger **`app.audit` hanya untuk superadmin.** Isinya email admin dan aksi
  pengelolaan akun, unit, konfigurasi, dan kill switch, yang selama ini ranah
  superadmin. Penyaringan dilakukan di API, bukan di UI: role `admin` tidak
  pernah menerima baris `app.audit` dari endpoint mana pun, termasuk lewat
  filter logger, pencarian teks, atau tautan `turn_id`.

### Kenapa SQLite, bukan Postgres

- Tidak bergantung pada Postgres. Log terpenting ("gagal mencatat percakapan ke
  database") justru hilang bila disimpan di Postgres yang sedang bermasalah.
- Volume besar, umur pendek: ±8–12 baris node per giliran. Di Postgres (yang di
  dev lewat SSH tunnel) itu berarti belasan write tambahan per pertanyaan.
- `sqlite3` ada di stdlib, jadi tanpa dependensi baru.
- API berjalan dengan satu worker uvicorn (asumsi yang juga dipakai kill switch
  dan rate limit), jadi satu file SQLite dengan mode WAL tidak berebut tulis.

### Konsekuensi yang diterima

- **Docker:** `api/log/` harus di-mount sebagai volume, karena filesystem
  container hilang setiap redeploy. Dengan pm2 (`start.sh`) tidak ada masalah.
- **Tidak bisa di-JOIN dengan Postgres.** Setiap baris membawa `turn_id`,
  `message_id`, dan `session_id` sebagai penghubung.
- **Multi-server:** tiap server punya log sendiri. Diterima untuk saat ini.

## Isi SQLite

Prinsip: **SQLite tidak menyimpan teks pertanyaan maupun jawaban.** Teks sudah
ada di Postgres, lengkap dengan penyamaran pertanyaan sensitif (FR-7). SQLite
hanya berisi metrik dan log, ditautkan lewat `message_id`.

### `turns`: satu baris per giliran chat

| Kolom | Contoh / arti |
|---|---|
| `turn_id` | UUID, dibuat di awal request |
| `timestamp` | awal giliran |
| `endpoint` | `chat` / `chat_stream` |
| `session_id`, `message_id` | untuk membuka percakapannya di Postgres |
| `unit` | unit pilihan mahasiswa, atau null |
| `outcome` | `answer`, `refusal`, `support`, `smalltalk`, `rejected` |
| `last_node` | titik keluar: `sensitive`, `smalltalk`, `jev_gate`, `refuse`, `generate` |
| `total_ms` | durasi total giliran |
| `ttft_ms` | waktu sampai token pertama (streaming saja) |
| `status` | `ok` / `error` / `dibatalkan` (mahasiswa menutup panel saat streaming) |
| `langsmith_run_id` | tautan ke trace bila tracing aktif |

### `node_runs`: satu baris per node per giliran

Kolom umum: `turn_id`, `position`, `node`, `started_at`, `duration_ms`, `status`,
`error_type`, `error_message` (dipotong 500 karakter), `detail` (JSON).

Isi `detail` per node:

| Node | `detail` |
|---|---|
| `sanitize` | `length`: panjang pertanyaan (angka saja) |
| `sensitive` | `level` sensitivitas, `redirected`: apakah alur dialihkan |
| `smalltalk` | `handled`: apakah pesan ditangani di sini |
| `jev_gate` | `disabled`, atau `label`, `confidence`, `blocked`, `cost_usd`, `error` fail-open |
| `rewrite` | `query_rewritten` (false juga saat dilewati; durasi ≈0 menandakan dilewati) |
| `retrieve` | `document_count` (unit filter ada di baris `turns`) |
| `validate_context` | `decision`, `reason` (`ok`/`no_results`/`below_threshold`), `top_score`, `top_rerank_score` |
| `refuse` | `contact_count`: jumlah kontak yang disarankan |
| `generate` | `model`, `input_tokens`/`output_tokens`, `cost_usd` |

### `app_logs`: semua log Python aplikasi

`timestamp`, `level`, `logger` (mis. `app.audit`, `app.rag.gate`,
`app.observability.chatlog`), `message`, `location` (modul:baris), `traceback`,
`turn_id` (terisi bila log muncul di tengah sebuah giliran, lewat contextvar).

### Retensi

Baris yang lebih tua dari 7 hari dihapus saat aplikasi start dan setiap jam.
Konfigurasi: `LOG_DB_PATH` (default `log/app.db`), `LOG_RETENTION_DAYS`
(default `7`).

## Jalur tulis

- **Node:** `AsyncCallbackHandler` dipasang lewat `config={"callbacks": [...]}`
  di `run_graph` (`app/rag/graph.py`). Nama node dibaca dari
  `metadata.langgraph_node`; router (`selesai_atau_*`, `route_context`) diabaikan.
  Tidak ada node yang perlu diubah.
- **Log Python:** konfigurasi logging di `lifespan` (level lewat env, format
  berisi timestamp, level, dan nama logger ke stdout) plus handler SQLite.
  Audit dikembalikan ke `info` setelah INFO benar-benar tercatat.
- **Tidak memblokir request:** record masuk `QueueHandler`, lalu satu thread
  `QueueListener` menulisnya ke SQLite secara batch. Kegagalan menulis log
  tidak boleh menggagalkan jawaban ke mahasiswa (prinsip yang sama dengan
  `chatlog.py`).

## Endpoint API (role minimal `admin`)

Admin tidak membaca file SQLite langsung; semua lewat API.

- `GET /api/admin/logs/summary?range=24h|7d`: angka KPI, p50/p95 per node,
  distribusi titik keluar, p95 per jam, error per jam.
- `GET /api/admin/logs/turns`: daftar giliran (filter hasil, unit, status; paginasi).
- `GET /api/admin/logs/turns/{turn_id}`: node satu giliran beserta `detail`.
- `GET /api/admin/logs/app`: log aplikasi (filter level, logger, rentang waktu,
  cari teks; paginasi). Untuk role `admin`, baris `app.audit` selalu dikecualikan
  di query SQL; meminta `logger=app.audit` menghasilkan daftar kosong, bukan 403,
  supaya keberadaan filternya tidak bocor lewat perbedaan respons. Angka error
  di `summary` juga dihitung tanpa `app.audit` untuk role ini.

## Halaman admin `/log`

Menu "Log" di grup yang sama dengan Statistik dan Biaya. Filter waktu
**24 jam / 7 hari** (7 hari = batas masa simpan). Grafik memakai komponen
recharts yang sudah ada (`admin/src/components/ui/chart.tsx`).

### Tab 1: Performa

```
┌ Giliran ─┐ ┌ p95 total ┐ ┌ Error ──┐ ┌ Diblokir JEV ┐
│   1.284  │ │  3,8 dtk  │ │  0,6 %  │ │    4,1 %     │
└──────────┘ └───────────┘ └─────────┘ └──────────────┘

Durasi per node (p50 ▓ / p95 ░)          Titik keluar giliran
generate         ▓▓▓▓▓▓▓▓▓▓░░░░░░ 2,9s    generate  ████████████ 71%
retrieve         ▓▓▓░░ 0,6s               refuse    ███ 14%
jev_gate         ▓▓░ 0,4s                 smalltalk █ 8%
rewrite          ▓▓░ 0,4s                 jev_gate  █ 4%
sanitize…valid.  ▏ <5ms                   sensitive ▏ 3%

p95 latensi per jam (garis, 7 hari) + jumlah error (batang)
```

### Tab 2: Giliran

```
Waktu     Hasil    Titik keluar  Total   Unit      Status
10:12:03  answer   generate      3,1 s   Keuangan  ok
10:11:40  refusal  refuse        0,9 s   —         ok
10:09:15  answer   generate      —       —         error
```

Klik baris untuk membuka panel detail berisi waterfall:

```
sanitize          ▏ 1ms
sensitive         ▏ 2ms
smalltalk         ▏ 1ms
jev_gate          ███ 380ms       academic · 0,97
rewrite           (dilewati)
retrieve             ████ 540ms   8 dokumen
validate_context         ▏ 1ms    answer · top 0,82
generate                 ████████████████ 2.150ms  gpt-5.5 · 1.240/310 tok
[Buka percakapan]  [Buka di LangSmith]
```

### Tab 3: Log aplikasi

Tabel log dengan filter level (INFO/WARNING/ERROR), filter logger, dan
pencarian teks. Pintasan filter: "Error saja", plus "Audit" yang hanya tampil
untuk superadmin. Baris ERROR bisa dibuka untuk melihat traceback dan tautan
ke gilirannya.

## Pertanyaan terbuka

Tidak ada. (Log audit untuk role `admin`: diputuskan disembunyikan, lihat
Keputusan.)

## Implementasi tahap 1

| Berkas | Isi |
|---|---|
| `api/app/observability/logstore.py` | skema SQLite, tulis batch, retensi, query dashboard |
| `api/app/observability/applog.py` | `LogWriter` (antrean + thread), `SQLiteLogHandler`, `configure_logging`, `NodeRecorder`, `catat_giliran` |
| `api/app/routers/admin_logs.py`, `api/app/schemas/logs.py` | endpoint `/api/admin/logs/*` |
| `api/app/main.py` | `mulai_log` di awal lifespan, `hentikan_log` saat berhenti |
| `api/app/routers/chat.py` | kedua endpoint chat dibungkus `catat_giliran`; TTFT di jalur streaming |
| `api/app/rag/graph.py`, `chain.py` | parameter `callbacks` diteruskan ke graf |
| `api/api.yaml` | tag `admin-logs`, empat operasi, skema, parameter `LogRange` |
| test | `tests/unit/test_logstore.py`, `tests/unit/test_applog.py`, `tests/api/test_admin_logs.py` |

Setelan baru: `LOG_DB_PATH`, `LOG_RETENTION_DAYS`, `LOG_LEVEL` (lihat `api/.env.example`).
Log ke konsol kini berformat `waktu LEVEL logger: pesan`, dan INFO tidak lagi hilang.
Kotak uji coba admin (`/api/admin/test-query`) sengaja tidak ikut tercatat.

## Urutan pengerjaan

1. ✅ API: modul penyimpanan SQLite + retensi, konfigurasi logging, callback node
   di `run_graph`, baris `turns` di router chat, endpoint, test (termasuk test
   bahwa role `admin` tidak pernah menerima baris `app.audit`).
2. ✅ `.gitignore` untuk `api/log/`, catatan volume Docker di `api/README.md`.
3. ✅ Admin: halaman `/log` (tiga tab) + item menu.

## Implementasi tahap 3

| Berkas | Isi |
|---|---|
| `admin/src/app/(dashboard)/log/page.tsx` | halaman, `RequireRole min="admin"` |
| `admin/src/components/logs/logs-view.tsx` | header, rentang 24 jam/7 hari, tiga tab, panel rincian bersama |
| `admin/src/components/logs/performance-tab.tsx` | KPI, durasi per langkah (p50/p95), titik keluar, grafik per jam |
| `admin/src/components/logs/turns-tab.tsx` | tabel giliran + filter hasil/status + paginasi |
| `admin/src/components/logs/turn-sheet.tsx` | panel rincian: ringkasan, waterfall langkah, log selama giliran |
| `admin/src/components/logs/app-logs-tab.tsx` | log aplikasi: cari, level, sumber, "Galat saja", "Audit admin" (superadmin) |
| `admin/src/lib/i18n/dict/logs.ts` | teks dua bahasa |
| `admin/src/lib/api/queries.ts` | `useLogSummary`, `useLogTurns`, `useLogTurn`, `useAppLogs` (diperbarui tiap menit) |

Beda dari wireframe:

- **"Buka percakapan" dan "Buka di LangSmith" menjadi ID yang dapat disalin.** Dashboard
  belum punya halaman percakapan, dan URL LangSmith butuh ID organisasi dan proyek yang
  tidak diketahui API. `message_id` dicari di tabel `messages`; ID trace ditempel di
  pencarian proyek LangSmith.
- **Durasi per langkah berupa batang per baris, bukan grafik recharts.** Selisih durasi
  antar langkah tiga orde (1 md vs beberapa detik); di satu sumbu linear langkah cepat
  tidak terlihat.
- Tren per jam dipecah menjadi dua grafik (waktu respons p95, dan galat) alih-alih satu
  grafik dua sumbu.
- Jam ditampilkan di zona waktu peramban; API menyimpan UTC.
