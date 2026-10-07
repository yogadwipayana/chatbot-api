# Tool Calling — Sumber Data Layanan Akademik

Dokumen desain untuk memberi LLM kemampuan memanggil **tool** (function calling)
agar dapat mengambil data dari layanan akademik SADS secara langsung saat
menjawab. Melengkapi `flow.md`: di sana LLM hanya meringkas konteks hasil
retrieval; di sini LLM boleh **meminta data terstruktur yang segar** lewat API
yang sudah terdaftar.

> **Status: terimplementasi & AKTIF di `.env` lokal (`TOOLS_ENABLED=true`).**
> Default kode tetap mati; `.env` lokal menyalakannya. Modul di `app/rag/tools/`,
> terintegrasi di node `generate` (`app/rag/graph.py`). Diuji
> `tests/unit/test_tools.py` (registry, validasi, handler, loop, integrasi) dan
> diverifikasi live lewat `POST /api/chat` (2026-10-07): "dosen pengampu mata
> kuliah Programming" dijawab lengkap dengan kartu sumber "Data akademik SADS"
> dan `meta.tool_calls` tercatat. Produksi: set `TOOLS_ENABLED=true` +
> `SADS_BASE_URL` + `SADS_API_SECRET` di env server.

Nama berkas di dalam `( )` relatif terhadap `api/`.

---

## 1. Kapan tool-calling, kapan ingestion

Tool-calling **bukan** pengganti indexing. Keduanya hidup berdampingan:

| | **Tool-calling** (dokumen ini) | **Ingestion** (`flow.md` §2) |
|---|---|---|
| Cocok untuk | data **dinamis / parametrik / personal** yang tak bisa dipra-materialisasi | dokumen & fakta **statis, dapat dienumerasi, ganti lambat** |
| Contoh | "jadwal **saya** hari ini", "status UKT **saya**", "sisa SKS **saya**", lookup parametrik dengan ruang nilai besar | panduan KRS, kode etik, FAQ, daftar yang kecil & tetap |
| Kesegaran | real-time tiap pertanyaan | sesegar sinkronisasi terakhir |
| Determinisme | LLM yang memutuskan memanggil (perlu penjagaan) | deterministik |
| Sitasi | kartu sumber sintetis ("Data SADS per <waktu>") | kartu dokumen asli |

Aturannya: **kalau datanya bisa dipra-materialisasi jadi beberapa chunk dan
jarang berubah, pakai ingestion.** Tool-calling dipilih di sini karena jumlah
endpoint layanan akademik akan terus bertambah (jadwal, nilai, pembayaran,
status mahasiswa) dan sebagian bersifat personal/real-time — kelas data yang
tidak mungkin diindeks di muka. Dua endpoint pertama (dosen & mata kuliah,
lihat §7) sebetulnya masih bisa diingest; keduanya dipakai sebagai **tool
perdana** untuk membangun dan menguji mekanismenya, bukan karena ingestion
mustahil untuk mereka.

---

## 2. Verifikasi gateway (2026-10-07)

Prasyarat mutlak: gateway (`BASE_URL`, OpenAI-compatible) dan `CHAT_MODEL`
(`cx/gpt-6-luna`) harus mendukung `tools` / `tool_choice`. Diuji langsung ke
`POST {BASE_URL}/chat/completions`:

| Uji | Hasil |
|---|---|
| Non-stream, `tool_choice:"auto"` | ✅ `finish_reason:"tool_calls"`, `content:null`, `tool_calls[0]` memanggil fungsi yang benar |
| Ekstraksi argumen | ✅ `{"matkul":"Programming"}` dari "dosen mata kuliah Programming" |
| Stream, `tools` | ✅ delta `tool_calls` standar OpenAI: potongan `arguments` di-concat (`{"` + `mat` + `kul` + `":"` + `Programming` + `"}`), ditutup `finish_reason:"tool_calls"` |
| `usage` | ada, **tetapi tidak konsisten** antara stream (`prompt_tokens` membengkak) dan non-stream |

**Kesimpulan:** tool-calling layak dibangun di gateway ini. Dua kuirk gateway
yang sudah dikenal tetap berlaku dan harus ditangani:

1. Gateway menyuntik `[Error] ... overloaded` sebagai **isi** stream
   (`providers.py`, `PENANDA_GALAT_GATEWAY`). Penyaring yang sama wajib dipakai
   di giliran jawaban akhir tool-loop.
2. Akunting token gateway tidak bisa dipercaya penuh — jangan jadikan dasar
   penagihan presisi pada jalur tool (FR-8/AD-5 tetap dicatat, dengan catatan
   ini).

---

## 3. Posisi di flow

Tool-calling **tidak** menambah node di awal pipeline. Ia mengubah node
`generate` (`flow.md` §9, `app/rag/graph.py::generate`) dari satu panggilan LLM
menjadi **loop agentik** singkat, dan menambah satu gerbang kelayakan sebelum
FR-3.

```text
          ... JEV academic → rewrite → retrieve → RRF → rerank ...
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ validate_context  │  (FR-3)
                         │ + tool_eligible?  │  ← BARU: §8
                         └─────────┬─────────┘
             konteks lemah &                 konteks cukup
             TIDAK tool-eligible                  atau tool-eligible
                     │                                 │
                     ▼                                 ▼
                  refuse                        ┌──────────────────┐
             (kontak unit)                      │ generate (loop)  │
                                                └────────┬─────────┘
                                                         ▼
                                   LLM(prompt + konteks, tools=terdaftar)
                                                         │
                            ┌────────────────────────────┴───────────┐
                     finish=tool_calls                         finish=stop
                            │                                         │
                            ▼                                         ▼
                 jalankan tiap tool (paralel)              stream jawaban → SSE
                 hasil → pesan role:"tool"                 (+ kartu sumber, §10)
                            │
                            ▼
                 panggil LLM lagi (maks. N putaran) ───────────────┘
```

Keterkaitan dengan gerbang lain tidak berubah:

- **FR-7 / smalltalk / rule_gate / JEV** tetap di depan. Pertanyaan tool yang
  sah bernilai `academic` di JEV, jadi lolos seperti biasa.
- **FR-3** tetap menjaga jalur RAG murni. Yang berubah: pertanyaan **tool-
  eligible** boleh maju ke `generate` walau konteks retrieval lemah (§8) —
  tanpa itu, "siapa dosen Web Programming" (tanpa dokumen PDF) akan ditolak
  sebelum tool sempat dipanggil.
- **FR-5** (pembungkusan `<pertanyaan_mahasiswa>`) dan **FR-6** (kontak unit
  pada topik berisiko) tetap berjalan di `generate`.

---

## 4. Komponen & berkas

| Komponen | Tugas | Berkas |
|---|---|---|
| Spesifikasi & primitif | `ToolSpec`, `ToolResult`, validasi argumen, kartu sitasi (§6, §10, §11) | `app/rag/tools/base.py` |
| Registry tool | Daftar tool + rute kelayakan `eligible()` | `app/rag/tools/registry.py` |
| Handler SADS | Panggil endpoint SADS + normalisasi hasil | `app/rag/tools/sads.py` |
| Klien HTTP | `httpx.AsyncClient`, header `secret`, timeout | `app/rag/tools/client.py` |
| Loop agentik | Orkestrasi LLM ↔ tool, streaming giliran final | `app/rag/tools/loop.py` |
| Binding tools ke LLM | `build_llm(...).bind(tools=…, tool_choice="auto")` + loop | `app/deps.py` (`LLMCall.run_tools`) |
| Gerbang kelayakan | Tandai `tool_eligible` sebelum FR-3, integrasi `generate` | `app/rag/graph.py` (`validate_context`, `route_context`, `generate`) |

Prinsip: **registry deklaratif, handler tipis.** Menambah endpoint = menambah
satu spesifikasi + satu handler + test — tanpa menyentuh loop atau graf (§16).

---

## 5. Loop agentik

Di dalam `generate`, ganti satu panggilan `llm_call` dengan loop:

```text
pesan = [system, konteks (bila ada), <pertanyaan_mahasiswa>]
untuk putaran 1..N:
    resp = LLM(pesan, tools=spec_yang_relevan, tool_choice="auto")
    bila resp.finish == "tool_calls":
        untuk tiap panggilan (boleh paralel):
            validasi argumen  (§11)
            hasil = handler(args)            # HTTP ke SADS
            pesan += {role:"tool", tool_call_id, content: hasil_ternormalisasi}
        lanjut
    selain itu:
        kembalikan resp.content              # jawaban final
kembalikan fallback "data tidak tersedia"    # batas putaran tercapai
```

- **Batas putaran** `N` kecil (mis. 2–3). Mencegah loop tak berujung bila model
  terus memanggil tool.
- **Panggilan paralel**: bila satu giliran memuat beberapa `tool_calls`,
  jalankan handler-nya bersamaan (`anyio`/`asyncio.gather`), masing-masing di
  klien HTTP sendiri — seperti FTS ∥ pgvector pada retrieval.
- **Bind hanya bila pertanyaannya lolos gerbang kelayakan** (§8), bukan pada
  setiap giliran akademik — skema tool ikut menambah token di tiap prompt. Tetapi
  begitu gerbang terbuka, **seluruh tool di-bind**, bukan hanya yang pemicunya
  cocok: pemilihan tool adalah tugas model lewat `description`. Pemicu yang tidak
  simetris kalau tidak justru menyembunyikan tool yang benar (lihat §18).

---

## 6. Kontrak tool

Satu tool dideklarasikan sekali, lalu dipakai untuk: (a) skema yang dikirim ke
LLM, (b) rute kelayakan, (c) eksekusi, (d) label sitasi.

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str                      # "get_mk_diampu_dosen"
    description: str               # dibaca LLM; jelas & sempit
    parameters: dict               # JSON Schema argumen
    handler: Callable[..., Awaitable[ToolResult]]
    triggers: tuple[str, ...]      # kata kunci rute kelayakan (§8)
    unit: str | None               # unit pemilik untuk kartu sumber
    citation_label: str            # "Data akademik SADS"
```

`ToolResult` membawa teks ternormalisasi (yang masuk ke pesan `role:"tool"`) dan
metadata untuk kartu sumber (label + waktu ambil). Skema `parameters` adalah
satu-satunya kontrak argumen — validasi di §11 menegakkannya sebelum handler
dipanggil.

---

## 7. Tool perdana (SADS)

Host `https://sads.instiki.ac.id`, auth **header `secret: <token>`** (bukan
Bearer, bukan query param — terverifikasi 2026-10-06). Token dari env
`SADS_API_SECRET`, base dari `SADS_BASE_URL`.

### `get_daftar_dosen`
- `GET /service/tp/chatbot/dosen-mengajar` — tanpa argumen.
- Respons: `[{"nmdosen": "..."}]`.
- Normalisasi: `.strip()` tiap `nmdosen` (data punya trailing space & koma).

### `get_mk_diampu_dosen`
- `GET /service/tp/chatbot/mk-diampu-dosen?matkul=<kata kunci>`
- Argumen: `matkul: string` (wajib).
- Respons: `[{"nmdosen":"...","matkul":[{"matkul":"..."}]}]` — dosen yang
  mengampu mata kuliah yang cocok kata kunci.
- Normalisasi: `.strip()` + ratakan jadi teks ringkas per baris.

> **Catatan enumerasi.** API `mk-diampu-dosen` tersusun **per dosen**. Untuk
> pertanyaan "sebutkan **semua** dosen mata kuliah X", hasil tool cukup karena
> satu panggilan mengembalikan seluruh dosen yang cocok — tidak seperti
> retrieval yang mengambil top-k. Inilah keunggulan konkret tool di sini.

---

## 8. Gerbang kelayakan & FR-3

Masalah: FR-3 menolak sebelum `generate` saat konteks retrieval lemah, sehingga
pertanyaan yang seharusnya dijawab tool tidak pernah sampai ke LLM.

Solusi (sejalan filosofi `rule_gate` — murah & dapat diaudit): **gerbang
kelayakan deterministik** sebelum FR-3.

```text
tool_eligible = ADA ToolSpec yang `triggers`-nya cocok pertanyaan bersih
                -> seluruh tool di-bind (ToolRegistry.eligible, §18)
```

Gerbang ini **tidak melihat unit pilihan mahasiswa**: data SADS bersifat
lintas-unit, dan kelayakan dihitung hanya dari teks pertanyaan. Unit tetap
memfilter retrieval RAG dan tetap diberitahukan ke LLM (baris `TOPIK_AKTIF`),
tetapi tidak menentukan ketersediaan tool.

Routing `validate_context` menjadi:

| Konteks retrieval | `tool_eligible` | Tindakan |
|---|---|---|
| cukup (FR-3 lolos) | — | `generate` (tools di-bind bila eligible) |
| lemah | ya | `generate` **tetap jalan**, tools di-bind |
| lemah | tidak | `refuse` (seperti sekarang) |

Dengan ini jaminan lama FR-3 utuh untuk semua pertanyaan non-tool, dan hanya
pertanyaan yang cocok pemicu tool yang boleh menembus konteks lemah. `triggers`
cukup berupa kata kunci/regex (mis. `dosen`, `mata kuliah`, `mengajar`,
`mengampu`), batal bila pertanyaan jelas di luar domain. Alternatif yang lebih
akurat (klasifikasi intent oleh model) ditunda — mulai dari aturan dulu, kalibrasi
dari log seperti JEV.

---

## 9. Streaming

Jalur sekarang (`chain.py::_jawab` → `deps.py::LLMCall.stream`) menstream
`potongan.text` langsung ke SSE lewat `on_token`, dengan `_PenahanPenanda`
menahan penanda `[TIDAK_DITEMUKAN]`/`[DI_LUAR_TOPIK]` di awal.

Yang harus berubah untuk tool-loop:

1. **Giliran tool_call tidak boleh di-stream sebagai jawaban.** Delta
   `tool_calls` (lihat §2) dikumpulkan di buffer, **bukan** diteruskan ke
   `on_token`. Selama ini berlangsung, panggil `on_stage("mengambil data
   akademik")` supaya indikator FE-1 tidak diam.
2. **Hanya giliran final** (`finish=stop`) yang di-stream token-nya ke
   `on_token`, lewat `_PenahanPenanda` dan `PenyaringGalatGateway` yang sama.
3. **`usage`** dijumlahkan lintas semua giliran LLM dalam loop (akunting gateway
   tidak presisi — §2).

`LLMCall` perlu mode baru yang menerima `tools` dan mengembalikan, per giliran,
entah `tool_calls` entah teks. `_jawab` tetap mengembalikan **jawaban utuh**
(sitasi FE-2 dan baris log FR-8 dibaca dari sana), jadi kontrak keluarannya tidak
berubah dari sudut pemanggil.

---

## 10. Sitasi hasil tool

Invarian sistem: tiap jawaban memetakan ke kartu sumber. Hasil tool bukan
dokumen, jadi dibuat **kartu sintetis**:

```text
Sumber: Data akademik SADS — diambil 2026-10-07 14:03
```

Kartu ini dibangkitkan dari `ToolResult` (label `citation_label` + waktu ambil),
bukan dari tabel `documents`. Tidak mencampur `DocumentType` (`pdf`,
`tanya_jawab`) — tool adalah jenis sumber ketiga di lapisan sitasi, bukan baris
dokumen baru. LLM tetap wajib menandai bagian jawaban yang bersumber tool, dan
hanya tool yang benar-benar dipakai yang jadi kartu (sejajar aturan sitasi PDF).

---

## 11. Keamanan

- **Injeksi lewat hasil tool.** Isi respons API diperlakukan sebagai **data,
  bukan instruksi**. Masuk sebagai pesan `role:"tool"`, dan system prompt
  menegaskan data tool tidak boleh mengubah aturan — senada FR-5 untuk input
  mahasiswa. Jangan pernah menaruh hasil tool ke dalam system prompt.
- **Validasi argumen (anti-SSRF & sampah).** Handler **tidak pernah** menerima
  URL dari LLM — hanya argumen bernama dari skema. Tiap argumen divalidasi ke
  skema `parameters` sebelum handler jalan: batasi panjang, buang karakter
  kontrol, tolak tipe salah. `matkul` yang dihalusinasi model paling buruk
  menghasilkan daftar kosong, bukan panggilan ke host lain.
- **Rahasia.** `secret` SADS hanya di env (`SADS_API_SECRET`), tak pernah
  di-log, tak pernah dikirim ke LLM, tak pernah masuk kartu sumber.
- **Timeout & batas.** Tiap panggilan tool punya timeout tegas; batas putaran
  loop (§5) dan batas jumlah tool per giliran mencegah penggandaan beban.

---

## 12. Galat & fallback

| Kegagalan | Tindakan |
|---|---|
| Tool HTTP galat / timeout | pesan `role:"tool"` = penanda `DATA_TIDAK_TERSEDIA`; LLM diminta jawab apa adanya atau akui tak tersedia |
| Model terus memanggil tool > N | hentikan loop, kembalikan fallback kontak unit (gaya refuse FR-3) |
| Gateway sisipkan `[Error]` di jawaban final | `PenyaringGalatGateway` menangkap, jawaban tidak tersimpan (seperti sekarang) |
| Model tak memanggil tool padahal perlu | jatuh ke jalur RAG biasa; bila konteks lemah → refuse. Bukan jawaban salah |

Prinsip sama seperti reranker/JEV: **tool memperkaya, bukan syarat menjawab.**
Kegagalan tool tidak boleh memunculkan jawaban ngawur.

---

## 13. Observability & biaya

- ✅ **`messages.meta.tool_calls`** (sejajar JEV): satu entri per panggilan tool
  — `name`, `args` (mentah dari model), `ok`, `latency_ms`. None bila tool tidak
  dipakai. Dialirkan loop → `LLMCall.tool_calls` → `catat` → `build_meta`
  (`app/observability/chatlog.py`). Contoh nyata (2026-10-07):
  `[{"name":"get_mk_diampu_dosen","args":{"matkul":"Programming"},"ok":true,"latency_ms":876}]`.
- Span LangSmith per giliran LLM dan per panggilan tool
  (`app/observability/tracing.py`).
- FR-8/AD-5: jumlahkan `usage` semua giliran; beri tanda bahwa angka gateway
  tidak presisi (§2).
- Latensi: harapkan ≥ 2 round-trip LLM untuk pertanyaan ber-tool. Pantau p95.

---

## 14. Konfigurasi (env)

```text
TOOLS_ENABLED=false            # sakelar utama; default mati saat rollout
TOOLS_MAX_ROUNDS=2             # batas putaran loop agentik
SADS_BASE_URL=https://sads.instiki.ac.id
SADS_API_SECRET=<secret>       # header `secret`
SADS_TIMEOUT_SECONDS=10
```

Mengikuti pola `RERANK_*`: fitur di belakang sakelar, default mati, nilai di
`config.py` + `.env.example`.

---

## 15. Testing

- **Registry/skema**: tiap `ToolSpec` punya JSON Schema valid; nama unik.
- **Handler** (HTTP di-mock): normalisasi benar (strip, ratakan); `matkul`
  diteruskan sebagai query; header `secret` terpasang.
- **Loop** (LLM di-mock memancarkan `tool_calls` lalu teks): urutan pesan benar,
  batas putaran dihormati, panggilan paralel jalan.
- **Streaming**: delta `tool_calls` tidak bocor ke `on_token`; hanya giliran
  final yang di-stream; `_PenahanPenanda` tetap menahan penanda.
- **Keamanan**: argumen cacat/karakter kontrol ditolak; hasil tool berisi
  "abaikan instruksi" tidak mengubah perilaku.
- **Fallback**: tool timeout → fallback; gateway `[Error]` di jawaban final
  tertangkap.
- **Kelayakan**: pertanyaan non-tool dengan konteks lemah tetap `refuse` (FR-3
  tidak bocor).

---

## 16. Checklist menambah endpoint

1. Tulis handler di `app/rag/tools/sads.py` (atau modul layanan lain): panggil
   endpoint, normalisasi ke teks ringkas, kembalikan `ToolResult`.
2. Deklarasikan `ToolSpec`: `name`, `description` (sempit & jelas),
   `parameters` (JSON Schema), `triggers`, `unit`, `citation_label`.
3. Daftarkan ke `registry.py`.
4. Tambah test (§15) untuk handler + skema.
5. Bila perlu env baru (base/secret layanan lain), tambah di `config.py` +
   `.env.example`.

Loop, streaming, sitasi, dan gerbang kelayakan **tidak** perlu disentuh.

---

## 17. Rencana bertahap

Semua fase sudah dijalankan (2026-10-07). `TOOLS_ENABLED=true` aktif di `.env`
lokal dan sudah diuji lewat `POST /api/chat` (jawaban + kartu sumber + `meta.tool_calls`).

1. ✅ **Fondasi** — `ToolSpec`, registry, klien HTTP, loop, di belakang
   `TOOLS_ENABLED=false`. Dua tool SADS perdana + test.
2. ✅ **Streaming** — giliran tool di-buffer (tak berkonten), hanya giliran
   jawaban final yang mengalir; penyaring `[Error]` + penahan penanda dipakai ulang.
3. ✅ **Integrasi gerbang** — `tool_eligible` sebelum FR-3 (`validate_context` +
   `route_context`), kartu sumber sintetis di `generate`.
4. ✅ **Observability** — `messages.meta.tool_calls` (nama/argumen/ok/latensi) +
   `usage` tercatat (§13). Lanjutan opsional: pantau latensi p95 produksi dan
   kalibrasi `triggers` dari log nyata.
5. ✅ **Nyalakan** — `TOOLS_ENABLED=true` aktif di `.env` lokal, terverifikasi
   lewat `/api/chat`. Untuk produksi: set `TOOLS_ENABLED=true` + `SADS_*` di env
   server.

---

## 18. Tinjauan 2026-10-07: yang diperbaiki & yang masih terbuka

### Diperbaiki

| Cacat | Akibat | Perbaikan |
|---|---|---|
| **Invarian FR-3 bocor.** `generate` dapat dicapai dengan vonis REFUSE (karena tool-eligible); bila tool ternyata tak dapat dijalankan (`llm_call` tanpa `run_tools`), ia memanggil LLM dengan KONTEKS kosong | satu panggilan LLM berbayar hanya untuk membalas `[TIDAK_DITEMUKAN]`, dan penolakannya tercatat `refusal_source = llm` padahal ambang yang menolak | `generate` mengembalikan penolakan FR-3 tanpa memanggil LLM bila `not pakai_tool` dan vonisnya REFUSE |
| **Pemicu tidak simetris.** `eligible()` hanya mem-bind tool yang pemicunya cocok | "ada berapa dosen?" hanya mem-bind `get_mk_diampu_dosen` (wajib `matkul`); `get_daftar_dosen` — satu-satunya yang bisa menjawab — tak terlihat, jadi model mengarang `matkul` atau menolak | `eligible()` membuka **seluruh** tool begitu ada satu pemicu cocok (§5, §8). Terverifikasi live: model kini memilih `get_daftar_dosen` dan menjawab "Terdapat 223 dosen … [Data akademik SADS]" |
| **Uji coba admin (AD-6) tanpa tool** | admin melihat `refusal` untuk pertanyaan yang dijawab mahasiswa — persis kekeliruan yang dicegah docstring-nya sendiri untuk JEV | `admin_quality.test_query` meneruskan `tool_registry`; `retrieved` tetap hanya chunk asli (sumber tool dilewati) |
| **Argumen model ditulis mentah ke `meta`** | string raksasa dari model menggelembungkan kolom `meta` setiap giliran | dipotong `MAKS_PANJANG_ARGUMEN` (`_args_untuk_log`) |
| **Field `ToolSpec.unit` mati** | menyiratkan scoping per-unit yang tidak ada | dibuang; alasannya didokumentasikan di `triggers` |

### Terverifikasi live (2026-10-07, setelah gateway pulih)

- **Jalur batas putaran** (`max_rounds` habis): giliran paksa lewat `llm_plain`
  mengirim riwayat yang memuat `role:"tool"` **tanpa** `tools` dideklarasikan.
  Gateway menerimanya dan menjawab normal — kekhawatiran gateway ketat tidak
  terbukti pada `cx/gpt-6-luna`.
- **Streaming** `/api/chat/stream`: stage `mengambil data akademik` muncul, tidak
  ada JSON `tool_calls` yang bocor ke token, dan token gabungan == teks final.

### Masih terbuka (sengaja tidak diubah tanpa keputusan)

1. **`eval/run_generation.py` tidak memakai tool.** Evaluasi RAGAS karena itu
   melihat `refusal` untuk pertanyaan yang di produksi dijawab tool. Tidak
   diwire supaya metodologi evaluasi yang sedang berjalan tidak berubah diam-diam.
2. **Penolakan jalur tool masuk AD-4** (`unanswered_questions`). Padahal itu
   bukan celah dokumen — menambah PDF tidak menyelesaikannya, dan dengan alasan
   yang sama `rejected` sengaja TIDAK masuk AD-4. Dapat dibedakan lewat
   `meta.tool_calls` yang terisi.
3. **Jawaban tool membawa `ThresholdDecision` REFUSE.** Akibatnya
   `messages.top_score` mendekati 0 untuk jawaban yang benar, dan `applog`
   mencatat `decision=refuse` pada node `validate_context` untuk giliran yang
   berakhir `answer`. Jujur apa adanya, tetapi perlu diingat saat membaca statistik.
4. **Asumsi giliran tool tanpa konten.** Gateway ini mengirim `content:null`
   pada giliran tool, jadi tidak ada token yang bocor. Model yang mengirim konten
   *dan* `tool_calls` dalam satu giliran akan membuat konten itu ikut mengalir
   sebelum jawaban final.
