# Skema Database

PostgreSQL 16 + [pgvector](https://github.com/pgvector/pgvector). **Satu** database
menampung metadata dokumen, vektor, indeks full-text, log percakapan, dan akun
dashboard — tidak ada Elasticsearch, tidak ada vector store terpisah (PRD §10).

Sumber kebenaran skema ada di dua tempat yang harus selalu sepadan:

| Berkas | Peran |
|---|---|
| `app/db/models.py` | Model SQLAlchemy 2.0 — dipakai kode dan `alembic check` |
| `alembic/versions/*.py` | Riwayat migrasi — yang benar-benar dijalankan di server |

Perubahan skema wajib menyentuh keduanya. `alembic check` akan gagal bila model
dan database berbeda.

---

## ERD

```mermaid
erDiagram
    documents ||--o{ chunks : "1..N (CASCADE)"
    conversations ||--o{ messages : "1..N (CASCADE)"
    messages ||--o{ feedback : "1..N (CASCADE)"
    messages ||--o| unanswered : "0..1 (SET NULL)"
    documents ||--o{ usage_log : "0..N (SET NULL)"
    units ||--o{ documents : "1..N (ON UPDATE CASCADE)"
    units |o--o{ admins : "0..N (ON UPDATE CASCADE)"

    units {
        varchar  nama PK "200 — nama resmi yang tampil di menu"
        varchar  deskripsi "500 — teks bantu menu chatbot"
        int      urutan "NOT NULL, default 0 — urutan tampil"
        boolean  is_active "NOT NULL, default true"
    }

    documents {
        uuid     id PK
        varchar  judul "500, NOT NULL — pertanyaan bila jenis=tanya_jawab"
        varchar  unit FK "200, NOT NULL → units.nama"
        varchar  jenis "20, NOT NULL, default 'pdf'"
        varchar  file_path "1000, NULL untuk tanya_jawab"
        varchar  nama_file "255, nama asli unggahan PDF"
        text     jawaban "NULL untuk pdf"
        int      tahun_berlaku
        date     valid_until "NULL = tanpa batas"
        varchar  uploaded_by "255"
        timestamptz updated_at "NOT NULL, default now()"
        boolean  is_active "NOT NULL, default true"
    }

    chunks {
        uuid     id PK
        uuid     document_id FK "NOT NULL"
        text     konten "NOT NULL"
        int      halaman "NOT NULL"
        int      urutan "NOT NULL"
        vector   embedding "1024 dim, NOT NULL"
        tsvector tsv "diisi trigger"
    }

    conversations {
        uuid     id PK
        varchar  session_id "128, NOT NULL"
        varchar  user_hash "64, anonim"
        timestamptz created_at "NOT NULL, default now()"
    }

    messages {
        uuid     id PK
        uuid     conversation_id FK "NOT NULL"
        varchar  role "20, NOT NULL — user/assistant"
        text     konten "NOT NULL"
        uuid_arr retrieved_chunk_ids "chunk yang dipakai"
        float    top_score "skor mentah tertinggi"
        int      latency_ms
        varchar  langsmith_run_id "64, akar trace; null bila tracing mati"
        jsonb    meta "kind, topik, token, biaya"
        timestamptz created_at "NOT NULL, default now()"
    }

    feedback {
        uuid     id PK
        uuid     message_id FK "NOT NULL"
        boolean  helpful "NOT NULL"
        text     catatan
        timestamptz created_at "NOT NULL, default now()"
    }

    unanswered {
        uuid     id PK
        text     pertanyaan "NOT NULL"
        float    top_score
        boolean  resolved "NOT NULL, default false"
        uuid     message_id FK "NULL-able, SET NULL"
        timestamptz created_at "NOT NULL, default now()"
    }

    runtime_config {
        varchar  key PK "64 — nama field Settings"
        varchar  value "64, NOT NULL — teks, sama seperti di .env"
        timestamptz updated_at "NOT NULL, default now()"
        varchar  updated_by "255 — email admin"
    }

    usage_log {
        uuid     id PK
        timestamptz created_at "NOT NULL, default now()"
        varchar  operasi "20, NOT NULL — ingest/reindex"
        varchar  model "200, NOT NULL — model yang diminta"
        varchar  model_dilaporkan "200, bila penyedia menyebut nama lain"
        int      tokens "NULL = tidak dilaporkan"
        float    biaya_usd "tanpa pembulatan"
        varchar  biaya_sumber "20 — provider/estimasi"
        boolean  is_byok
        uuid     document_id FK "NULL-able, SET NULL"
        varchar  keterangan "500 — judul dokumen saat itu"
    }

    admins {
        uuid     id PK
        varchar  email "255, UNIQUE + unik lower()"
        varchar  password_hash "255, NOT NULL"
        varchar  role "50, NOT NULL — staf/admin/superadmin"
        varchar  nama "200"
        varchar  unit FK "200 → units.nama, wajib untuk staf"
        boolean  is_active "NOT NULL, default true"
        timestamptz password_changed_at "pembatal token lama"
        timestamptz last_login_at
        timestamptz created_at "NOT NULL, default now()"
    }
```

`runtime_config` berdiri sendiri tanpa relasi: satu baris per parameter `.env` yang
ditimpa dari dashboard, dan **hanya** parameter yang benar-benar ditimpa. Tidak
ada barisnya berarti "ikut `.env`" -- lihat `app/admin/runtime_config.py`.

`admins` hanya berelasi ke `units`. Jejak admin pada dokumen disimpan sebagai
teks di `documents.uploaded_by`, bukan foreign key: menghapus akun tidak boleh
menghapus atau membuat NULL riwayat dokumen yang pernah diunggahnya.

Dua CHECK constraint di `admins` menegakkan level akses di database:

```sql
CONSTRAINT ck_admins_role      CHECK (role IN ('staf', 'admin', 'superadmin'))
CONSTRAINT ck_admins_staf_unit CHECK (role <> 'staf' OR (unit IS NOT NULL AND btrim(unit) <> ''))
```

### Kelompok tabel

| Kelompok | Tabel | Ditulis oleh | Dibaca oleh |
|---|---|---|---|
| **Pengetahuan** | `documents`, `chunks` | `app/ingestion/`, `app/admin/faq.py` | retrieval FR-2 |
| **Referensi** | `units` | migrasi `0009` / SQL manual — belum ada API untuk mengubahnya | `app/units.py`: menu chatbot, validasi setiap isian unit, filter retrieval |
| **Log** | `conversations`, `messages`, `feedback`, `unanswered` | `app/observability/chatlog.py`, `app/routers/chat.py` | dashboard AD-4, AD-5 |
| **Akun** | `admins` | `app/admin/accounts.py`, `scripts/create_admin.py` | auth AD-1 |
| **Setelan** | `runtime_config` | `app/routers/admin_config.py` | `get_effective_settings` di setiap permintaan |
| **Biaya** | `usage_log` | belum ada (lihat status di bawah) | biaya AD-5 |

---

## `documents` — satu tabel, dua jenis isi

Baris `documents` adalah **sumber kebenaran yang dapat disunting**; `chunks`
hanyalah turunannya. Kolom `jenis` menentukan dari mana isinya berasal:

| `jenis` | Asal isi | `file_path` | `nama_file` | `jawaban` | `judul` |
|---|---|---|---|---|---|
| `pdf` | Berkas resmi yang diunggah admin | kunci objek, **NOT NULL** | nama asli unggahan | **NULL** | judul dokumen |
| `tanya_jawab` | Diketik admin di dashboard | **NULL** | **NULL** | isi jawaban, **NOT NULL** | pertanyaannya |

Keduanya berbagi satu tabel supaya seluruh mesin yang sudah ada berlaku tanpa
perubahan: filter dokumen aktif, masa berlaku, unit, chunking, embedding, dan
sitasi. Retrieval tidak perlu tahu bedanya.

Aturan itu ditegakkan database, bukan hanya kode:

```sql
CONSTRAINT ck_documents_jenis CHECK (jenis IN ('pdf', 'tanya_jawab'))

CONSTRAINT ck_documents_isi_sesuai_jenis CHECK (
     (jenis = 'pdf'         AND file_path IS NOT NULL AND jawaban IS NULL)
  OR (jenis = 'tanya_jawab' AND file_path IS NULL     AND jawaban IS NOT NULL)
)
```

Alasannya konkret: PDF tanpa berkas membuat kartu sitasi FE-2 menunjuk ke
ketiadaan — dan itu terlihat mahasiswa; entri tanya jawab tanpa jawaban tidak
dapat disunting kembali.

**`file_path` menyimpan kunci objek** (`documents/<uuid>.pdf`), bukan lintasan
disk. Berpindah dari disk lokal ke S3/R2 karena itu tidak memaksa migrasi data.
**`nama_file` menyimpan nama asli unggahan** untuk `Content-Disposition` saat
PDF dibuka atau diunduh. Nama ini tidak dipakai sebagai kunci objek, sehingga
nama yang sama dari dua unggahan tidak saling menimpa.

### Dua predikat turunan `documents`

Keduanya punya satu definisi saja di kode, dan sengaja saling berlawanan:

| Predikat | Modul | Arti |
|---|---|---|
| `active_document_clause()` | `app/rag/filters.py` | `is_active AND (valid_until IS NULL OR valid_until > now())` — **wajib** ada di setiap query retrieval |
| `stale_clause()` | `app/admin/documents.py` | `updated_at < now() - interval '6 months' OR valid_until <= now()` — badge "perlu ditinjau" (AD-2) |

Batas `valid_until` di keduanya persis berlawanan, sehingga dokumen yang diberi
badge kedaluwarsa di dashboard tepat dokumen yang sudah berhenti terambil
retrieval. Penyaringan terjadi di dalam `WHERE` SQL, bukan setelah hasil kembali
— dokumen kedaluwarsa tidak boleh pernah ikut terambil lalu disaring belakangan.

---

## `units` — daftar tetap unit layanan

Mahasiswa memilih unit di menu chatbot, lalu retrieval hanya mencari di dokumen
unit itu. Filter tersebut hanya dapat dipercaya bila setiap dokumen memakai nama
yang **persis** sama — selama `documents.unit` diketik bebas, dokumen berlabel
"Bagian Keuangan" tidak pernah terambil untuk pilihan "Keuangan", dan mahasiswa
menerima penolakan padahal jawabannya ada. Karena itu `documents.unit` dan
`admins.unit` kini foreign key ke `units.nama`.

**Nama sebagai kunci utama, bukan kode terpisah.** Nilai yang tersimpan di
`documents.unit` tetap nama yang tampil di dashboard, sehingga kontrak API admin
tidak berubah. `ON UPDATE CASCADE` membuat penggantian nama cukup satu `UPDATE`
di `units`; dokumen dan akun staf ikut.

**Unit tidak dihapus, tetapi dinonaktifkan.** `ON DELETE` memakai bawaan
`NO ACTION`: menghapus unit yang masih dipakai ditolak database. `is_active =
false` menyembunyikannya dari menu (`GET /api/units`) dan dari pilihan untuk
dokumen baru; dokumen lamanya tetap ada.

**Normalisasi di pintu masuk, perbandingan persis di retrieval.** Setiap isian
unit (unggah dokumen, entri tanya jawab, akun, chat, uji coba) melewati
`deps.unit_terdaftar` → `app/units.py::cocokkan`, yang mencocokkan lewat
`permissions.normalize_unit` dan mengembalikan ejaan resmi — `" keuangan "`
menjadi `Keuangan`, nama tak dikenal ditolak 422. Filter retrieval sendiri
membandingkan persis:

```sql
(CAST(:unit AS text) IS NULL OR d.unit = CAST(:unit AS text))   -- NULL = semua unit
```

Satu SQL untuk kedua kasus, bukan dua varian query yang bisa menyimpang. `CAST`
wajib: asyncpg tidak dapat menebak tipe parameter yang hanya muncul di `IS NULL`.

**Memecah `chunks` per unit sempat dipertimbangkan dan ditolak.** Pertanyaan
lintas unit (cuti akademik menyangkut BAAK dan Keuangan) menjadi `UNION`, indeks
HNSW dan GIN berlipat sembilan, dan memindah unit sebuah dokumen berarti
memindah seluruh chunk-nya antar-tabel.

Isi awal (migrasi `0009`, dalam urutan menu): BAAK, FO (Front Office), Keuangan,
Kemahasiswaan, Prodi, Fakultas, PLK, UPS (Unit Pelayanan Sertifikasi), Akademik.

---

## `chunks` — vektor dan full-text di baris yang sama

Satu baris = satu potongan teks siap di-embed. Ditulis lewat SQL langsung
(`app/ingestion/embedder.py`), bukan `vectorstore.add_documents()`, karena kolom
metadata kustom dan pembuatan `tsvector` tidak terjangkau abstraksi VectorStore.

| Kolom | Catatan |
|---|---|
| `embedding vector(1024)` | Dimensinya **harus** sama dengan keluaran `EMBED_MODEL` |
| `tsv tsvector` | Diisi trigger, bukan kode aplikasi |
| `halaman` | Selalu `1` untuk entri tanya jawab — sitasi berformat `[Judul, hal. N]` |
| `urutan` | Urutan chunk dalam dokumen, dipakai untuk merangkai konteks |

`EMBEDDING_DIM = 1024` muncul di `app/db/models.py` dan `alembic/versions/0001`.
**Mengganti `EMBED_MODEL` ke model berdimensi lain bukan sekadar migrasi kolom** —
seluruh dokumen harus di-index ulang. `embed_and_store` memeriksa dimensi
sebelum `INSERT` supaya galatnya menyebut `EMBED_MODEL`, bukan `DBAPIError`
generik dari pgvector.

### Trigger `tsv`

```sql
CREATE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('indonesian', COALESCE(NEW.konten, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_chunks_tsv
    BEFORE INSERT OR UPDATE OF konten ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update();
```

Diisi otomatis supaya tidak ada jalur penulisan yang bisa lupa mengisinya dan
diam-diam mematikan separuh retrieval hibrida.

> **Prasyarat rilis:** konfigurasi FTS `indonesian` harus tersedia di instans
> target. Periksa dengan `SELECT cfgname FROM pg_ts_config;`. Bila tidak ada,
> seluruh jalur FTS gagal diam-diam. Test integrasi memeriksa ini.

---

## Indeks

| Indeks | Tabel | Definisi | Untuk |
|---|---|---|---|
| `ix_documents_aktif` | `documents` | `(is_active, valid_until)` | filter dokumen aktif FR-2 |
| `ix_documents_unit` | `documents` | `(unit)` | filter unit retrieval, daftar dokumen staf |
| `ix_chunks_document_id` | `chunks` | `(document_id)` | JOIN retrieval, hitung chunk per dokumen |
| `ix_chunks_tsv` | `chunks` | **GIN** `(tsv)` | full-text search |
| `ix_chunks_embedding_hnsw` | `chunks` | **HNSW** `(embedding vector_cosine_ops)` | vector search |
| `ix_conversations_session_id` | `conversations` | `(session_id)` | mencari percakapan aktif satu sesi |
| `ix_messages_conversation_id` | `messages` | `(conversation_id)` | merangkai riwayat |
| `ix_messages_created_at` | `messages` | `(created_at)` | rentang tanggal AD-5 |
| `ix_feedback_message_id` | `feedback` | `(message_id)` | agregasi kepuasan, JOIN daftar umpan balik |
| `ix_unanswered_created_at` | `unanswered` | `(created_at)` | filter `sejak` AD-4 |
| `ix_admins_email_lower` | `admins` | **UNIQUE** `(lower(email))` | login tidak peka huruf besar |
| `ix_usage_log_created_at` | `usage_log` | `(created_at)` | rentang bulan halaman Biaya AD-5 |

`ix_documents_unit` dibuat terpisah karena foreign key di Postgres **tidak**
otomatis ber-indeks; tanpa itu filter unit menjadi sequential scan.

Dua indeks yang mudah rusak tanpa terasa:

**HNSW — operator class harus cocok.** `vector_cosine_ops` sepadan dengan
operator `<=>` yang dipakai `app/rag/retriever.py`. Indeks dengan operator class
lain akan diabaikan optimizer **secara diam-diam**: hasil pencarian tetap benar,
tetapi berubah menjadi sequential scan. Indeks ini dibuat lewat SQL mentah di
migrasi, **dan** tetap dideklarasikan di `models.py` — tanpa deklarasi itu,
`alembic revision --autogenerate` akan menganggapnya indeks liar dan menghasilkan
`DROP INDEX`.

**HNSW + filter unit — butuh iterative scan.** HNSW menyaring *setelah* indeks
dipindai, dan `hnsw.ef_search` bawaannya 40: bila satu unit hanya ~1/9 isi
indeks, rata-rata cuma 4–5 dari 40 kandidat yang lolos filter. Karena itu
pencarian vektor yang difilter unit menjalankan
`SET LOCAL hnsw.iterative_scan = strict_order` lebih dulu
(`retriever.ITERATIVE_SCAN_SQL`). Pencarian tanpa filter unit tidak menyentuhnya.

> **Prasyarat rilis:** `hnsw.iterative_scan` ada sejak **pgvector 0.8**. Periksa
> dengan `SELECT extversion FROM pg_extension WHERE extname = 'vector';`.

**`ix_admins_email_lower`.** Login dan pencarian akun memakai `lower(email)`.
Tanpa indeks unik ini, `Admin@instiki.ac.id` dan `admin@instiki.ac.id` bisa menjadi
dua akun berbeda.

---

## Perilaku penghapusan

| Relasi | Aksi | Alasan |
|---|---|---|
| `chunks.document_id` → `documents.id` | `CASCADE` | Chunk yatim tetap terambil retrieval tanpa induk dokumen yang sah |
| `messages.conversation_id` → `conversations.id` | `CASCADE` | Pesan tanpa percakapan tidak punya arti |
| `feedback.message_id` → `messages.id` | `CASCADE` | Umpan balik tanpa pesan tidak dapat ditafsirkan |
| `unanswered.message_id` → `messages.id` | **`SET NULL`** | Log percakapan boleh dibersihkan, sinyal perbaikan AD-4 tidak ikut hilang |
| `usage_log.document_id` → `documents.id` | **`SET NULL`** | Menghapus dokumen tidak boleh mengecilkan laporan biaya bulan yang sudah lewat |
| `documents.unit` → `units.nama` | **`NO ACTION`**, `ON UPDATE CASCADE` | Unit yang masih dipakai tidak boleh hilang — nonaktifkan lewat `is_active` |
| `admins.unit` → `units.nama` | **`NO ACTION`**, `ON UPDATE CASCADE` | Sama; `unit` NULL tetap sah untuk admin/superadmin |

`unanswered` sengaja berbeda. Ia bukan turunan log, melainkan daftar pekerjaan
admin — retensi log tidak boleh mengosongkannya.

---

## `messages.meta` (JSONB)

Ditulis `app/observability/chatlog.py::build_meta`, dibaca `app/admin/stats.py`.
Hanya diisi pada baris `role = 'assistant'`.

| Kunci | Tipe | Isi |
|---|---|---|
| `kind` | string | `answer` \| `refusal` \| `support` \| `smalltalk` |
| `escalated` | bool | Jawaban menyertakan kontak unit (FR-6) |
| `topik` | array | Topik berisiko tinggi yang terdeteksi |
| `sensitivitas` | string \| null | Tingkat sensitif (FR-7) |
| `llm_dipanggil` | bool | Penolakan FR-3 dan FR-7 bernilai `false` |
| `model` | string \| null | Hanya diisi bila LLM benar-benar dipanggil |
| `input_tokens` / `output_tokens` | int \| null | Dari `usage_metadata` LangChain |
| `biaya_usd` | float \| null | `null` bila tarif modelnya tidak dikenal |
| `rewritten_query` | string \| null | Hasil penulisan ulang query (FR-4) |
| `unit` | string \| null | Unit pilihan mahasiswa di menu; `null` = semua unit. Membedakan penolakan akibat salah pilih unit dari dokumen yang memang belum ada |
| `embed_dipanggil` | bool | `false` untuk FR-7 dan smalltalk — keduanya berhenti sebelum retrieval |
| `embed_model` | string \| null | Model yang **diminta**, bukan yang dilaporkan gateway |
| `embed_tokens` | int \| null | `usage.prompt_tokens`; `null` bila endpoint tidak melaporkannya |
| `embed_biaya_usd` | float \| null | Biaya meng-embed pertanyaan, **tanpa pembulatan** |
| `embed_biaya_sumber` | string \| null | `provider` atau `estimasi` |

Struktur ini bukan sekadar catatan — statistik AD-5 memfilter langsung atasnya
(`m.meta->>'kind'`, `m.meta->'topik'`). Menambah nilai `kind` baru tanpa
menyesuaikan `app/admin/stats.py` membuat pesan itu hilang dari semua hitungan.

**`biaya_usd = null` ≠ biaya nol.** Jawaban dari model yang tarifnya belum ada di
`app/observability/costs.py` dilaporkan terpisah sebagai
`pesan_tanpa_estimasi_biaya`, bukan dianggap gratis.

**`biaya_usd` adalah biaya LLM saja, bukan total.** Biaya embedding berdiri di
kunci `embed_biaya_usd` dan sengaja tidak dijumlahkan ke dalamnya: `biaya_usd`
sudah berarti "biaya LLM" di seluruh baris yang tercatat sebelumnya dan di setiap
query `app/admin/stats.py`. Mengubah artinya diam-diam membuat baris sebelum dan
sesudah perubahan tidak lagi sebanding. Penjumlahan keduanya dilakukan saat
menyajikan, bukan saat menyimpan.

**Penolakan FR-3 berbiaya embedding meskipun `llm_dipanggil = false`.** Retrieval
berjalan lebih dulu (`app/rag/chain.py:141`), baru ambangnya memutuskan menolak —
pertanyaannya sudah terlanjur di-embed. Karena itu penjumlahan biaya embedding
**tidak boleh** menumpang filter `meta->>'llm_dipanggil' = 'true'` yang dipakai
query biaya LLM; pakai syaratnya sendiri, `meta->>'embed_tokens' IS NOT NULL`.
Menyaringnya dengan filter yang salah akan menghapus seluruh penolakan dari
laporan — padahal pertanyaan yang banyak ditolak justru yang paling perlu terlihat
(sinyal AD-4).

Bedakan tiga keadaan: `embed_dipanggil = false` berarti benar-benar tidak ada
panggilan (FR-7 dan smalltalk); `true` dengan `embed_tokens = null` berarti
panggilannya nyata dan berbiaya tetapi endpoint tidak melaporkan pemakaian; `true`
dengan angka lengkap berarti terhitung penuh. Hanya yang pertama yang gratis.

---

## `usage_log` — biaya yang tidak punya baris pesan

Biaya chat menumpang `messages.meta`. Embedding saat ingestion tidak bisa: ia
terjadi ketika tidak ada mahasiswa yang bertanya sama sekali, jadi tidak ada
baris yang dapat dititipi. Tanpa tabel ini halaman Biaya AD-5 diam-diam hanya
melaporkan sebagian dari yang benar-benar dibelanjakan.

Kolom di `documents` sempat dipertimbangkan dan ditolak. Menghapus dokumen akan
mengecilkan total bulan yang sudah lewat, dan menyunting entri tanya jawab akan
menimpa biaya ingestion pertamanya — buku biaya yang berubah surut tidak dapat
menjawab "bulan lalu habis berapa". Karena itu tabelnya **append-only** dan
`document_id` memakai `SET NULL`, alasan yang sama dengan `unanswered.message_id`.
`keterangan` menyimpan judul dokumen saat panggilan terjadi supaya barisnya tetap
terbaca setelah induknya hilang.

### Biaya penyedia mengalahkan taksiran

Gateway proyek ini mengembalikan `usage.cost` — biaya sebenarnya, bukan hitungan
kita. Angka itu selalu menang; `costs.PRICES_PER_MTOK` hanyalah salinan tarif yang
bisa tertinggal, dan dipakai hanya untuk endpoint yang tidak melaporkan biaya
(`cost` bukan bagian spesifikasi OpenAI). `biaya_sumber` mencatat yang mana yang
terpakai, ditegakkan database:

```sql
CONSTRAINT ck_usage_log_biaya_lengkap CHECK ((biaya_usd IS NULL) = (biaya_sumber IS NULL))
```

Angka biaya tanpa asal-usul tidak dapat ditafsirkan lagi setelah beberapa bulan,
dan asal-usul tanpa angka tidak ada artinya.

### `biaya_usd` disimpan tanpa pembulatan

Ini bukan kerapian, melainkan syarat agar kolomnya berguna. `costs.estimate_cost`
membulatkan ke enam desimal, sementara embedding satu pertanyaan (sembilan token
pada `text-embedding-3-small`) berharga **$0,00000018** — `round(…, 6)`
menjadikannya nol bulat. Ribuan pertanyaan yang seluruhnya berbiaya nol bukan
laporan biaya. Karena itu jalur embedding memakai `costs.estimate_input_cost`
yang tidak membulatkan, dan pembulatan baru terjadi di `app/admin/stats.py`
setelah dijumlahkan.

> **Status:** embedding pertanyaan mahasiswa sudah tercatat di `messages.meta`
> dan ditampilkan halaman Biaya. `app/admin/stats.py` juga sudah membaca
> `usage_log`, tetapi tabel itu **belum punya penulis** — `app/ingestion/` dan
> `app/admin/faq.py` belum mengisinya. Karena itu bagian ingestion/reindex masih
> nol sampai jalur penulisannya diterapkan.

---

## Yang tidak disimpan, dan mengapa

Ini bagian skema yang paling mudah dilanggar tanpa sadar (PRD §11):

- **Identitas mahasiswa.** `conversations.user_hash` adalah hash anonim dan tidak
  boleh dapat dikembalikan ke identitas. `session_id` berasal dari localStorage
  peramban, bukan dari NIM.
- **Isi pertanyaan sensitif (FR-7).** `messages.konten` untuk pertanyaan sensitif
  diganti penanda tetap: `[disembunyikan: pertanyaan sensitif, dialihkan ke
  layanan konseling]`. Jumlahnya tetap tercatat untuk statistik, tetapi curahan
  hati mahasiswa tidak ikut terbaca siapa pun yang membuka database.
- **Uji coba admin (AD-6)** tidak dicatat sama sekali.
- **Kill switch dan pembatas login** tinggal di memori proses, bukan tabel. Benar
  untuk satu worker uvicorn (sesuai Dockerfile); harus dipindah ke Postgres bila
  jumlah worker ditambah.

---

## Pola penulisan yang perlu dipertahankan

**Ingestion satu transaksi.** Dokumen + seluruh chunk-nya masuk dalam satu
transaksi (`app/ingestion/pipeline.py`). Dokumen yang gagal di tengah jalan tidak
boleh meninggalkan indeks setengah terisi.

**Berkas diunggah sebelum baris DB ditulis.** Penyimpanan objek berada di luar
transaksi database, jadi salah satunya pasti bisa gagal sendirian. Urutannya
dipilih agar kegagalan menghasilkan objek yatim (hanya memakan tempat), bukan
baris yatim (kartu sitasi menunjuk ke ketiadaan — terlihat mahasiswa). Bila
transaksi DB gagal, objeknya dihapus lagi.

**Suntingan tanya jawab memicu re-index.** Mengubah `judul` atau `jawaban` pada
entri `tanya_jawab` membuang chunk lama dan menghitung ulang embedding. Indeks
yang masih memuat kalimat versi lama akan menjawab mahasiswa dengan aturan yang
sudah dicabut.

**`updated_at` hanya bergerak untuk perubahan isi.** `is_active` tidak termasuk:
menyalakan ulang dokumen lama tidak boleh menghapus badge "perlu ditinjau"-nya.

**Log gagal ditulis tidak menggagalkan jawaban.** `ChatLogger.log` menelan
galatnya dan mengembalikan `None`. Database yang tersendat sesaat lebih baik
kehilangan satu baris log daripada membuat mahasiswa melihat galat setelah
jawabannya sudah jadi.

**`clock_timestamp()`, bukan `now()`, untuk `messages`.** `now()` bernilai sama
sepanjang transaksi, sehingga pertanyaan dan jawabannya akan bercap waktu
identik dan urutannya tak tentu.

**Percakapan dipotong oleh jeda 30 menit.** `session_id` tinggal di localStorage
berbulan-bulan; tanpa batas jeda (`CONVERSATION_IDLE_MINUTES`), seluruh pemakaian
satu peramban terhitung sebagai satu percakapan.

**Statistik dikelompokkan menurut `TIMEZONE`, bukan UTC.** Default
`Asia/Jakarta` — supaya pertanyaan pukul 06.00 WIB tidak tercatat sebagai
pertanyaan hari sebelumnya.

---

## Riwayat migrasi

| Revisi | Isi |
|---|---|
| `0001_skema_awal` | Extension `vector`, tujuh tabel, indeks HNSW + GIN, trigger `tsv` |
| `0002_selaraskan_not_null` | Tujuh kolom NOT NULL yang tertinggal — termasuk `chunks.embedding` |
| `0003_log_dashboard_admin` | `unanswered.message_id`, indeks `created_at` untuk AD-4/AD-5 |
| `0004_level_akses_admin` | `nama`, `unit`, `is_active`, `password_changed_at`, `last_login_at`; `editor` → `admin`; unik `lower(email)` |
| `0005_entri_tanya_jawab` | `jenis`, `jawaban`; `file_path` menjadi nullable; dua CHECK constraint |
| `0006_nama_file_asli` | `nama_file` untuk nama tab browser dan nama unduhan PDF |
| `0007_konfigurasi_runtime` | `runtime_config` — parameter `.env` yang dapat ditimpa dari dashboard |
| `0008_buku_biaya_pemakaian` | `usage_log` — biaya embedding yang tidak punya baris pesan untuk ditumpangi |
| `0009_tabel_unit` | `units` + isi awal; nilai `unit` lama dipetakan ke nama resmi; FK dari `documents`/`admins`; `ix_documents_unit` |

Catatan per migrasi:

- **0002** aman hanya selama tabel belum berisi NULL pada kolom tersebut.
  `chunks.embedding` NULL membuat chunk tidak pernah terambil vector search —
  dokumen tampak terindeks padahal separuh retrieval tidak melihatnya.
- **0005 downgrade menghapus data.** Baris `tanya_jawab` tidak punya berkas, jadi
  tidak ada cara mengubahnya menjadi dokumen PDF yang sah — baris beserta
  chunk-nya dihapus.
- **0009 gagal, bukan menebak, untuk nilai unit tak dikenal.** Nilai lama
  dipetakan lewat `ALIAS` di berkas migrasi ("Bagian Keuangan" → `Keuangan`).
  Nilai yang tidak cocok membuat migrasi berhenti dengan daftar nilainya —
  perbaiki dengan `UPDATE documents/admins SET unit = ...` atau tambahkan ke
  `ALIAS`, lalu jalankan ulang. Dokumen yang diam-diam masuk unit yang salah
  justru hilang dari menu unit yang benar.
- **0009 downgrade tidak mengembalikan nama lama.** Pemetaan ke nama resmi
  tetap berlaku; nilai aslinya tidak disimpan di mana pun.

```bash
alembic upgrade head          # terapkan
alembic check                 # model vs database harus sepadan
alembic revision --autogenerate -m "..."
alembic downgrade -1
```

## Koneksi

```
postgresql+asyncpg://<user>:<pass>@<host>:5432/<db>
```

Driver **asyncpg** (`app/db/session.py`), pool default 10 (`DB_POOL_SIZE`),
`pool_pre_ping=True`.

Dua konsekuensi asyncpg yang menyentuh skema:

**Tipe `vector` tidak punya codec asyncpg.** Embedding dikirim sebagai literal
teks yang di-cast `::vector` di SQL — mengirim list Python langsung gagal dengan
`expected str, got list`. Formatnya dibentuk satu fungsi saja,
`retriever.vector_literal`, dan dipakai baik saat menulis maupun saat mencari;
dua format berbeda adalah sumber bug yang sulit dilacak.

**Satu koneksi menolak dua query bersamaan** (`another operation is in
progress`). Karena itu `PostgresHybridRetriever` membuka sesi terpisah untuk
pencarian vektor dan full-text — paralelisme FR-2 hanya mungkin dengan dua
koneksi.

Akses langsung ke database produksi lewat SSH tunnel: lihat `docs/start.md`.
