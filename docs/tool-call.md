# Tool Calling — Sumber Data Layanan Akademik

Dokumen ini menjelaskan cara LLM memanggil **tool** (function calling) untuk
mengambil data layanan akademik SADS secara langsung saat menjawab. Ia
melengkapi `flow.md`. Di sana LLM meringkas konteks hasil retrieval; di sini LLM
boleh **meminta data terstruktur yang segar** lewat API yang terdaftar.

> **Status: terimplementasi, AKTIF di `.env` lokal (`TOOLS_ENABLED=true`).**
> Bawaan kodenya tetap mati. Modulnya di `app/rag/tools/`, terintegrasi di
> `app/rag/graph.py` (`validate_context`, `route_context`, `generate`). Diuji
> `tests/unit/test_tools.py` dan diverifikasi live lewat `POST /api/chat` dengan
> seluruh komponen asli: retriever (dev DB), JEV, rewrite, LLM, dan SADS (§18).
> Untuk produksi, set `TOOLS_ENABLED=true`, `SADS_BASE_URL`, dan
> `SADS_API_SECRET` di env server.

Nama berkas di dalam `( )` relatif terhadap `api/`.

---

## 1. Kapan tool-calling, kapan ingestion

Tool-calling **bukan** pengganti indexing. Keduanya hidup berdampingan:

| | **Tool-calling** (dokumen ini) | **Ingestion** (`flow.md` §2) |
|---|---|---|
| Cocok untuk | data **dinamis / parametrik / personal** yang tak bisa dipra-materialisasi | dokumen & fakta **statis, dapat dienumerasi, ganti lambat** |
| Contoh | "jadwal **saya** hari ini", "status UKT **saya**", lookup parametrik dengan ruang nilai besar | panduan KRS, kode etik, FAQ |
| Kesegaran | real-time tiap pertanyaan | sesegar unggahan terakhir |
| Determinisme | LLM yang memutuskan memanggil (perlu penjagaan) | deterministik |
| Sitasi | kartu sintetis "Data akademik SADS", tanpa tautan (§10) | kartu dokumen asli |

Aturannya: **kalau datanya bisa dipra-materialisasi menjadi beberapa chunk dan
jarang berubah, pakai ingestion.** Tool-calling dipilih karena jumlah endpoint
layanan akademik akan terus bertambah (jadwal, nilai, pembayaran, status
mahasiswa), dan sebagian bersifat personal atau real-time. Data seperti itu tidak
mungkin diindeks di muka. Dua endpoint pertama (dosen dan mata kuliah, §7)
sebetulnya masih bisa diingest. Keduanya dipakai sebagai **tool perdana** untuk
membangun dan menguji mekanismenya.

---

## 2. Verifikasi gateway (2026-10-07)

Prasyarat mutlak: gateway (`BASE_URL`, OpenAI-compatible) dan `CHAT_MODEL`
(`cx/gpt-6-luna`) mendukung `tools` / `tool_choice`. Diuji langsung ke
`POST {BASE_URL}/chat/completions`:

| Uji | Hasil |
|---|---|
| Non-stream, `tool_choice:"auto"` | ✅ `finish_reason:"tool_calls"`, `content:null`, fungsi yang dipanggil benar |
| Ekstraksi argumen | ✅ `{"matkul":"Programming"}` dari "dosen mata kuliah Programming" |
| Stream | ✅ delta `tool_calls` standar OpenAI: potongan `arguments` di-concat, ditutup `finish_reason:"tool_calls"` |
| Riwayat berisi `role:"tool"` **tanpa** `tools` dideklarasikan | ✅ diterima (dipakai saat batas putaran habis, §5) |
| `usage` | ada, **tetapi tidak konsisten** antara stream (`prompt_tokens` membengkak) dan non-stream |

Dua kuirk gateway yang sudah dikenal tetap berlaku:

1. Gateway menyuntik `[Error] ... overloaded` sebagai **isi** stream
   (`providers.py`, `PENANDA_GALAT_GATEWAY`). Penyaring yang sama dipakai di
   setiap giliran loop tool.
2. Akunting token gateway tidak bisa dipercaya penuh. FR-8/AD-5 tetap dicatat,
   tetapi jangan dijadikan dasar penagihan presisi.

---

## 3. Posisi di flow

Tool-calling **tidak** menambah node di pipeline. Ia menambah gerbang kelayakan
di `validate_context` dan mengubah `generate` dari satu panggilan LLM menjadi
**loop agentik** singkat.

```text
   ... JEV academic → rewrite → retrieve → RRF → rerank ...
                            │
                            ▼
                  ┌────────────────────┐
                  │ validate_context   │  FR-3 + tool_eligible (§8)
                  └─────────┬──────────┘
     konteks lemah DAN            konteks cukup ATAU
     bukan tool-eligible          tool-eligible
              │                          │
              ▼                          ▼
           refuse                 ┌──────────────────┐
      (kontak unit,               │ generate (loop)  │
       tanpa LLM)                 └────────┬─────────┘
                                           ▼
                        LLM(prompt + konteks, tools = seluruh registry)
                                           │
                    ┌──────────────────────┴───────────────┐
             finish = tool_calls                     finish = stop
                    │                                      │
                    ▼                                      ▼
     jalankan tool (paralel, maks. 4)             stream jawaban → SSE
     hasil → pesan role:"tool"                    (+ kartu sumber, §10)
                    │
                    ▼
     LLM lagi, maks. TOOLS_MAX_ROUNDS;
     lewat batas → jawaban dipaksa tanpa tool ─────────────┘
```

Gerbang lain tidak berubah:

- **FR-7 / smalltalk / rule_gate / JEV** tetap di depan. Pertanyaan tool yang sah
  dilabeli `academic` oleh JEV, jadi lolos seperti biasa (§18 mencatat
  keyakinannya).
- **FR-3** tetap menjaga jalur RAG murni. Yang berubah hanya satu: pertanyaan
  **tool-eligible** boleh maju ke `generate` walau konteks retrieval lemah (§8).
- **FR-5** (pembungkusan `<pertanyaan_mahasiswa>`) dan **FR-6** (kontak unit pada
  topik berisiko) tetap berjalan di `generate`.

---

## 4. Komponen & berkas

| Komponen | Tugas | Berkas |
|---|---|---|
| Spesifikasi & primitif | `ToolSpec`, `ToolResult`, validasi argumen, kartu sitasi | `app/rag/tools/base.py` |
| Registry | Daftar tool + gerbang kelayakan `eligible()` | `app/rag/tools/registry.py` |
| Handler SADS | Panggil endpoint SADS, normalisasi, hitung jumlah | `app/rag/tools/sads.py` |
| Klien HTTP | `httpx.AsyncClient`, header `secret`, timeout, `transport` untuk test | `app/rag/tools/client.py` |
| Loop agentik | LLM ↔ tool, eksekusi paralel, batas, streaming | `app/rag/tools/loop.py` |
| Binding ke LLM | `build_llm(...).bind(tools=…, tool_choice="auto")` | `app/deps.py` (`LLMCall.run_tools`) |
| Kelayakan & routing | `tool_eligible`, penolakan FR-3 bila tool tak tersedia, pertanyaan untuk loop | `app/rag/graph.py` (`_tool_eligible`, `route_context`, `_pesan_tool`, `generate`) |
| Aturan prompt | Aturan alat T1–T6 disisipkan sebelum KONTEKS | `app/rag/prompts.py` (`TOOL_SYSTEM_PROMPT`) |
| Lampiran | Daftar hasil tool tampil langsung di widget, per 10 baris (§10a) | `app/rag/tools/base.py` (`Lampiran`), `app/rag/graph.py` (`generate`), `ChatResponse.attachments`, `client/src/components/chat/chat-message.tsx` (`Lampiran`) |
| Pencatatan | `messages.meta.tool_calls`, `messages.meta.attachments` | `app/observability/chatlog.py` |

Prinsipnya: **registry deklaratif, handler tipis.** Menambah endpoint berarti
menambah satu `ToolSpec`, satu handler, dan test-nya, tanpa menyentuh loop
maupun graf (§16).

Selain `/api/chat` dan `/api/chat/stream`, uji coba admin (`POST
/api/admin/test-query`, AD-6) juga meneruskan registry, supaya admin melihat
jawaban yang sama dengan mahasiswa. Tabel `retrieved` di sana tetap hanya berisi
chunk hasil retrieval.

---

## 5. Loop agentik

`LLMCall.run_tools` → `run_tool_loop` (`app/rag/tools/loop.py`):

```text
pesan = [TOOL_SYSTEM_PROMPT + KONTEKS, <pertanyaan_mahasiswa> (§8)]
untuk putaran 1..TOOLS_MAX_ROUNDS:
    resp = LLM(pesan, tools = seluruh registry, tool_choice = "auto")   # di-stream
    bila resp tanpa tool_calls dan tanpa invalid_tool_calls:
        kembalikan resp.content            # jawaban final (kosong → [TIDAK_DITEMUKAN])
    on_stage("mengambil data akademik")
    jalankan ≤ 4 tool_calls BERSAMAAN (asyncio.gather):
        validasi argumen (§11) → handler → hasil sebagai pesan role:"tool"
    panggilan ke-5 dst. → pesan role:"tool" "batas … tercapai", tidak dijalankan
    invalid_tool_calls → pesan role:"tool" "argumen tidak dapat dibaca"
    on_stage("menyusun jawaban")
jawaban dipaksa lewat LLM TANPA tool                # batas putaran habis
```

- **Batas putaran** `TOOLS_MAX_ROUNDS` (bawaan 2, rentang 1–5) mencegah loop tak
  berujung. Setelah batas, giliran terakhir dipanggil tanpa `tools` sehingga
  model pasti menjawab dengan teks.
- **Paralel**: tool dalam satu giliran berjalan bersamaan, masing-masing dengan
  klien HTTP sendiri, seperti FTS ∥ pgvector. Urutan pesan `role:"tool"`
  mengikuti urutan panggilan.
- **Batas per giliran** `MAKS_TOOL_PER_GILIRAN = 4`. Setiap `tool_call_id` tetap
  dibalas tepat satu pesan `role:"tool"`. Tanpa balasan itu API menolak giliran
  berikutnya.
- **Seluruh tool di-bind** begitu gerbang terbuka, bukan hanya tool yang
  pemicunya cocok. Memilih tool adalah tugas model lewat `description` (§8).
- **Satu `run_id` per giliran LLM** (`buat_config`). Setiap giliran mendapat
  config baru dari `LLMCall._config`, sehingga semua giliran tercatat di
  LangSmith. `LLMCall.run_id` berakhir pada giliran jawaban.

---

## 6. Kontrak tool

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str                      # "get_mk_diampu_dosen"
    description: str               # dibaca LLM; jelas & sempit
    parameters: dict               # JSON Schema argumen (gaya OpenAI)
    handler: Callable[..., Awaitable[ToolResult]]
    triggers: tuple[str, ...]      # kata kunci gerbang kelayakan (§8)
    citation_label: str            # "Data akademik SADS"

@dataclass(frozen=True)
class ToolResult:
    name: str
    label: str                     # penanda sitasi; kosong bila gagal
    text: str                      # teks ternormalisasi untuk model
    ok: bool = True
    attachment: Lampiran | None = None   # daftar untuk widget (§10a)

@dataclass(frozen=True)
class Lampiran:
    title: str                     # "Dosen bergelar Dr."
    source: str                    # penanda sitasi asal datanya
    items: tuple[str, ...]
```

`ToolResult.pesan_untuk_model()` membentuk isi pesan `role:"tool"`. Bila
berhasil, isinya `(Kutip data berikut dengan menyalin penanda [label].)` diikuti
datanya; hasil berlampiran mendapat sisipan `DAFTAR_DITAMPILKAN` (§10a). Bila
gagal, isinya `DATA_TIDAK_TERSEDIA: …`. Skema `parameters` adalah
satu-satunya kontrak argumen; `validasi_argumen` menegakkannya sebelum handler
dipanggil.

`ToolSpec` sengaja tidak punya field unit. Kelayakan tool tidak bergantung pada
unit pilihan mahasiswa (§8).

---

## 7. Tool SADS

Host `https://sads.instiki.ac.id`, auth **header `secret: <token>`**, bukan
Bearer dan bukan query param (terverifikasi 2026-10-06). Base dari
`SADS_BASE_URL`, token dari `SADS_API_SECRET`.

### `get_daftar_dosen`
- `GET /service/tp/chatbot/dosen-mengajar`. Argumen opsional `nama` dan `gelar`
  disaring di handler; endpoint SADS-nya sendiri tanpa parameter.
- Respons: `[{"nmdosen": "..."}]`.
- Normalisasi: buang spasi dan koma di ujung, buang duplikat, urutkan.
- Saringan (`_cocok`, `_urai_dosen`):
  - `nama` dicocokkan sebagai bagian **nama inti**, tanpa beda huruf besar-kecil
    dan spasi ganda. Nama inti adalah teks sebelum koma pertama tanpa gelar
    depan (`Prof`, `Dr`, `Drs`, `Dra`, `Ir`), jadi "kom" mencocokkan "Komang",
    bukan gelar "S.Kom".
  - `gelar` dicocokkan sebagai **token utuh** setelah titik dan spasi dibuang
    ("Dr." = "dr" = "DR"), dari gelar depan maupun belakang (dipisah koma atau
    spasi). "Dr." tidak mencocokkan "Drs.". Sinonim: "doktor" = Dr. atau
    Ph.D., "profesor" = Prof.
  - Keduanya boleh digabung (dan).
- Teksnya diawali **jumlah yang dihitung handler** ("Jumlah dosen yang mengajar
  di INSTIKI bergelar Dr.: 25 orang."), ditambah jumlah seluruh dosen bila
  disaring. LLM tidak andal menghitung atau menyaring daftar panjang. Pada
  2026-10-07, dari 220 nama, model menjawab "223 dosen"; dari 25 dosen bergelar
  Dr., "38" lalu "39"; dan "dosen bernama Wayan" kehilangan 1 dari 15 (T46).
- Saringan tanpa hasil adalah jawaban sah ("Tidak ada dosen … bergelar Prof.:
  0 orang"): `ok=True`, tanpa lampiran. SADS yang tidak mengembalikan satu nama
  pun tetap `ok=False` (§12).
- Hasil yang tidak kosong membawa **lampiran** (§10a) berisi nama-nama yang
  cocok, berjudul mis. "Dosen bergelar Dr.".

### `get_mk_diampu_dosen`
- `GET /service/tp/chatbot/mk-diampu-dosen?matkul=<kata kunci>`.
- Argumen: `matkul: string` (wajib), satu mata kuliah per panggilan.
- Respons: `[{"nmdosen":"...","matkul":[{"matkul":"..."}]}]`, yaitu dosen yang
  mengampu mata kuliah yang cocok dengan kata kunci. SADS mencocokkan `matkul`
  sebagai **potongan teks persis**.
- Normalisasi: ratakan menjadi `- Nama: MK1, MK2`, plus jumlah dosen.
- **Varian ejaan** (`_varian_ejaan`): kata kunci berhuruf ganda juga ditanyakan
  dengan huruf ganda dirapatkan, lalu hasilnya digabung per dosen. SADS menulis
  "Artificial Intelligence" (34 dosen) dan "Artificial Inteligence" (1 dosen).
- **Tidak ada yang cocok bukan galat** (T43): `ok=True`, "Tidak ada mata kuliah
  di SADS yang namanya memuat "…": 0 dosen pengampu.". Dulu dibalas
  `DATA_TIDAK_TERSEDIA` ("tidak dapat diambil saat ini"), sehingga model mengira
  layanannya gangguan dan menolak dengan kontak FO.
- **Nama Inggris** (T47): sebagian nama mata kuliah di SADS berbahasa Inggris
  ("Artificial Intelligence", "Algorithms", "Database"). `description` meminta
  model mencoba padanan Inggris, sinonim, atau kata kunci yang lebih pendek bila
  hasilnya 0, dan menegaskan bahwa pencarian ini menurut nama mata kuliah, bukan
  nama dosen. Petunjuk itu sengaja di `description` (dipercaya), bukan di hasil
  tool (data, aturan T2). Aturan T6 menyatakan bahwa menjawab dari padanan
  seperti itu bukan menebak; tanpa T6 aturan 3 membuat model menjawab "belum
  dapat dipastikan apakah mata kuliah tersebut sama" tanpa satu nama pun.

### `get_mk_dosen`
- `GET /service/tp/chatbot/mk-diampu-dosen` **tanpa** `matkul`: SADS
  mengembalikan semua dosen beserta seluruh mata kuliahnya (220 dosen, 2.309
  pasangan, sekitar 98 KB, 1–3 dtk; uji 2026-10-07). Endpoint-nya tidak bisa
  disaring menurut dosen, jadi saringan dikerjakan handler.
- Argumen: `nama: string` (wajib), dicocokkan ke nama inti seperti saringan
  `nama` `get_daftar_dosen` (`_cocok`). Sapaan di depan (Pak, Bapak, Bu, Ibu)
  dibuang.
- Satu dosen cocok: daftar mata kuliahnya (urut abjad) beserta jumlahnya, dan
  **lampiran** "Mata kuliah yang diampu <nama>". Dua sampai
  `MAKS_DOSEN_DIRINCI` (5) dosen: dirinci tanpa lampiran. Lebih dari itu ("Pak
  Wayan" = 15 dosen): hanya nama-namanya, supaya model meminta nama lengkap.
- Tidak ada yang cocok: `ok=True`, "… 0 orang (dari N dosen …)". SADS kosong:
  `ok=False`.
- Ada karena T48: "mata kuliah apa yang diajar Ahmad Asroni?" dulu membuat model
  mengisi `get_mk_diampu_dosen(matkul="Ahmad Asroni")`, lalu menjawab "belum
  dapat diambil saat ini" atau menolak.

> **Catatan enumerasi.** Satu panggilan mengembalikan seluruh dosen yang cocok,
> tidak seperti retrieval yang mengambil top-k. Pertanyaan "sebutkan **semua**
> dosen mata kuliah X" karena itu terjawab lengkap.

---

## 8. Gerbang kelayakan & FR-3

Masalahnya: FR-3 menolak sebelum `generate` saat konteks retrieval lemah,
sehingga pertanyaan yang seharusnya dijawab tool tidak pernah sampai ke LLM.

Solusinya sejalan dengan filosofi `rule_gate`, murah dan dapat diaudit:

```text
tool_eligible = ADA ToolSpec yang `triggers`-nya cocok (substring, tanpa beda
                huruf besar-kecil) dengan pertanyaan bersih ATAU hasil rewrite
              → seluruh tool di-bind
```

- **Hasil rewrite ikut dicek** (`_tool_eligible`). Pertanyaan lanjutan ("kalau
  Basis Data?") tidak memuat pemicu, tetapi versi mandirinya ("Siapa dosen
  pengampu mata kuliah Basis Data?") memuatnya.
- **Loop menerima versi mandiri itu** (`_pesan_tool`). Loop tidak menerima
  riwayat percakapan, jadi pertanyaan asli dan hasil rewrite dikirim bersama,
  **keduanya di dalam** `<pertanyaan_mahasiswa>`. Hasil rewrite juga berasal dari
  teks mahasiswa, jadi diperlakukan sebagai data (FR-5).
- **Pemicu Inggris** (`lecturer`, `teach`). Rewrite hanya berjalan bila ada
  riwayat atau pertanyaan terdeteksi berbahasa Inggris, dan detektornya butuh
  minimal dua kata tugas Inggris. Akibatnya "who teaches Web Programming?" tidak
  di-rewrite, sehingga pemicunya harus mengenal kata Inggris sendiri.
- **Seluruh tool di-bind.** Pemicu yang tidak simetris pernah menyembunyikan tool
  yang benar: "ada berapa dosen?" hanya cocok dengan pemicu `get_mk_diampu_dosen`,
  yang mewajibkan `matkul`.
- **Tidak melihat unit pilihan.** Data SADS bersifat lintas-unit. Unit tetap
  memfilter retrieval RAG dan tetap diberitahukan ke LLM lewat baris
  `TOPIK_AKTIF`.

Routing (`route_context`, `generate`):

| Konteks retrieval | `tool_eligible` | Tindakan |
|---|---|---|
| cukup (FR-3 lolos) | tidak | `generate`, jalur RAG biasa |
| cukup | ya | `generate` dengan loop tool |
| lemah | ya | `generate` dengan loop tool |
| lemah | ya, tetapi tool tak dapat dijalankan | penolakan FR-3 **tanpa** memanggil LLM |
| lemah | tidak | `refuse`, seperti sebelum tool ada |

Jaminan FR-3 tetap utuh untuk semua pertanyaan non-tool. Alternatif yang lebih
akurat, yaitu klasifikasi intent oleh model, ditunda. Pemicu dikalibrasi dari
log `meta.tool_calls`, seperti ambang JEV.

---

## 9. Streaming

`_satu_giliran` men-stream setiap giliran LLM. Kontennya melewati
`PenyaringGalatGateway` (penanda `[Error]`) dan `_PenahanPenanda` (penanda
`[TIDAK_DITEMUKAN]`/`[DI_LUAR_TOPIK]`), sama seperti jalur jawaban biasa.

- **Giliran tool tidak memancarkan token.** Gateway ini mengirim `content:null`
  pada giliran tool, jadi delta `tool_calls` tidak pernah sampai ke `on_token`.
  Terverifikasi lewat SSE: tidak ada JSON tool-call di token, dan gabungan token
  sama dengan teks final.
- **Status SSE** pada jalur tool: `mencari dokumen` → `menyusun jawaban` →
  `mengambil data akademik` (saat tool berjalan) → `menyusun jawaban` → token.
  Frontend menampilkan teks `stage` apa adanya, jadi tahap baru tidak butuh
  perubahan klien.
- `run_tools` tetap mengembalikan **jawaban utuh**. Sitasi FE-2, deteksi
  penanda, dan log FR-8 dibaca dari sana, seperti `_jawab`.

---

## 10. Sitasi hasil tool

Hasil tool bukan baris di tabel `documents`. Agar mesin sitasi yang ada dapat
dipakai ulang, `hasil_tool_ke_dokumen` membuat **satu Document semu per label**
untuk tool yang berhasil:

```text
judul = "Data akademik SADS", jenis = tanya_jawab, halaman = 1,
file_path = "", document_id = "", dari_tool = True, tanpa chunk_id
```

- `jenis` `tanya_jawab` membuat pipeline memperlakukannya sebagai sumber tak
  berhalaman (penanda `[Judul]`). `dari_tool` membuat `citations_for` memberi
  kartunya `CitationOut.type` **`data`**: widget menampilkannya tanpa tautan dan
  tanpa "hal. N", berikon basis data dan berlabel "Data langsung", bukan "Tanya
  jawab resmi" (T44). Tidak ada stempel waktu pengambilan.
- Model mengutip `[Data akademik SADS]` (aturan T3). `citations_for` membuat
  kartu hanya bila penanda itu benar-benar dikutip.
- `is_not_found` dan `is_off_topic` menerima daftar sumber tak berhalaman dari
  `generate`. Tanpa itu `[Data akademik SADS]` tidak terbaca sebagai sitasi, dan
  jawaban parsial ("Basis Data diampu Budi [Data akademik SADS]. Untuk Kalkulus
  tidak ditemukan.") dibuang menjadi penolakan. Perbaikan ini juga berlaku untuk
  entri tanya jawab yang dikutip `[Judul]`.
- Document semu tidak punya `chunk_id`, jadi tidak masuk
  `messages.retrieved_chunk_ids` maupun tabel `retrieved` di uji coba admin.

---

## 10a. Lampiran: daftar langsung ke widget

Daftar panjang dari tool tidak ditulis ulang model. Uji 2026-10-07: daftar 220
dosen yang disalin model memang akurat, tetapi memakan sekitar 3,8 ribu token
keluaran dan 27–70 detik menulis, dan di widget menjadi gelembung setinggi 17
layar. Begitu disalin, setiap hitungan atau saringan atasnya juga dikerjakan
model (T46).

Alurnya:

1. Handler mengisi `ToolResult.attachment = Lampiran(title, source, items)`.
2. `pesan_untuk_model` tetap mengirim datanya utuh, karena model butuh nama-
   namanya untuk menjawab "apakah Pak Totok dosen INSTIKI?". Sisipan
   `DAFTAR_DITAMPILKAN` (`CATATAN_LAMPIRAN`) memberi tahu bahwa daftarnya sudah
   tampil, dan memuat petunjuknya sendiri: jangan menyalin seluruh daftar, salin
   jumlahnya, jawab ringkas. Aturan T5 hanya mengesahkan penanda itu sebagai
   pesan sistem (pengecualian T2). Petunjuk yang sama pernah ditaruh di T5 dan
   bocor ke tool lain: "siapa dosen pengampu Web Programming?" dijawab
   "berjumlah 23 orang" tanpa nama (§18).
3. `run_tool_loop` mengumpulkan lampiran dari tool yang berhasil
   (`ToolLoopResult.attachments` → `LLMCall.attachments`).
4. `generate` meneruskannya ke `PipelineOutcome.attachments` **hanya** untuk
   `answer` dan hanya bila `source`-nya dikutip jawaban, aturan yang sama dengan
   kartu sitasi (`citations_for`). Model yang memanggil tool tetapi tidak
   memakai hasilnya tidak menempelkan daftar 220 dosen di bawah jawaban lain.
5. `ChatResponse.attachments` (juga di event SSE `message` dan
   `TestQueryResponse`) membawa `{title, source, items}`. Widget menampilkannya
   di bawah jawaban, sebelum kartu Sumber, **10 baris per halaman** dengan
   tombol ‹ › dan nomor yang berlanjut antarhalaman. Uji coba admin menampilkan
   kotak yang sama. Seperti sitasi, lampiran tidak ditampilkan bila aliran SSE
   putus sebelum `done`.
6. `messages.meta.attachments` menyimpan lampiran utuh (§13).

Batasan:

- Lampiran tidak ikut riwayat percakapan. `history` yang dikirim widget hanya
  berisi `text`, jadi "yang nomor 11 siapa?" tidak dapat dijawab dari lampiran
  sebelumnya.
- `get_daftar_dosen` dan `get_mk_dosen` (satu dosen) berlampiran.
  `get_mk_diampu_dosen` sengaja belum: SADS mencocokkan `matkul` sebagai
  substring, dan model yang menyaring hasilnya ("Basis Data" tanpa "Basis Data
  Lanjut"). Lampiran mentah akan berbeda dari jawaban model. Akibatnya daftar
  panjang tetap disalin model: "Kecerdasan Buatan" (35 dosen) dari 5 percobaan
  2 lengkap, 2 kehilangan 1 nama, dan 1 hanya menyebut 4 nama ("antara lain").
- `CATATAN_LAMPIRAN` dulu membolehkan "nama tertentu bila pertanyaannya tentang
  orang itu". Untuk `get_mk_dosen` setiap pertanyaan memang tentang satu orang,
  sehingga 25 mata kuliah Ahmad Asroni disalin lengkap di atas lampiran yang
  sama. Kini hanya butir yang ditanyakan secara khusus yang boleh disebut
  ("apakah Pak Totok mengajar Basis Data?").

---

## 11. Keamanan

- **Injeksi lewat hasil tool.** Respons API masuk sebagai pesan `role:"tool"`,
  tidak pernah ke system prompt. Aturan T2 menegaskan isinya adalah data, bukan
  perintah.
- **Validasi argumen (anti-SSRF).** Handler **tidak pernah** menerima URL dari
  model, hanya argumen bernama dari skema. `validasi_argumen` membuang argumen
  tak dikenal, membersihkan karakter non-cetak, memotong string ke 200 karakter,
  dan menolak argumen wajib yang kosong. Argumen string bernilai `null` dibuang
  (berarti "tidak disaring"), bukan diteruskan sebagai teks "None". `matkul` yang dihalusinasi paling buruk
  menghasilkan daftar kosong.
- **Rahasia.** `secret` SADS hanya ada di env (`SADS_API_SECRET`). Ia tidak
  pernah di-log, dikirim ke LLM, atau masuk kartu sumber. Aplikasi menolak start
  bila `TOOLS_ENABLED=true` tetapi `SADS_BASE_URL` atau `SADS_API_SECRET` kosong.
- **Batas beban.** Timeout `SADS_TIMEOUT_SECONDS` per panggilan, maksimal 4 tool
  per giliran, dan `TOOLS_MAX_ROUNDS` putaran.

---

## 12. Galat & fallback

| Kegagalan | Tindakan |
|---|---|
| Tool HTTP galat / timeout / respons tak terduga | pesan `role:"tool"` = `DATA_TIDAK_TERSEDIA`; model menjawab apa adanya atau menolak |
| Saringan `get_daftar_dosen` tanpa hasil | `ok=True`, "Tidak ada dosen …: 0 orang", tanpa lampiran; model menjawab "tidak ada" bersumber SADS |
| `get_mk_diampu_dosen` tanpa hasil | `ok=True`, "… 0 dosen pengampu"; model mencoba padanan Inggris atau kata kunci lain (§7) |
| `get_mk_dosen` tanpa hasil / nama terlalu umum | `ok=True`, "0 orang" / hanya nama-nama dosen, tanpa lampiran |
| Argumen tool bukan JSON sah (`invalid_tool_calls`) | dibalas "argumen tidak dapat dibaca", loop berlanjut; tercatat `error = argumen_rusak` |
| Lebih dari 4 tool dalam satu giliran | sisanya dibalas "batas tercapai" tanpa dijalankan; `error = batas_per_giliran` |
| Model terus memanggil tool sampai batas putaran | giliran terakhir dipanggil tanpa `tools`, jadi model pasti menjawab teks |
| Giliran final tanpa teks | diganti `[TIDAK_DITEMUKAN]` → penolakan resmi beserta kontak, bukan `answer` kosong |
| Gateway menyisipkan `[Error]` | `GalatGateway` dilempar, giliran gagal dan tidak ada jawaban yang disimpan (seperti jalur biasa) |
| Tool tak dapat dijalankan, konteks lemah | penolakan FR-3 tanpa LLM (`refusal_source = threshold`) |

Prinsipnya sama seperti reranker dan JEV: **tool memperkaya, bukan syarat
menjawab.** Kegagalan tool tidak boleh memunculkan jawaban ngawur.

---

## 13. Observability & biaya

- **`messages.meta.tool_calls`**: satu entri per panggilan, `{name, args, ok,
  latency_ms}`. Panggilan yang tidak dijalankan juga membawa `error`. `args`
  adalah argumen mentah dari model, dengan setiap string dipotong 200 karakter.
  Nilainya `null` bila tidak ada tool yang dipanggil. Alirannya loop →
  `LLMCall.tool_calls` → `catat` → `build_meta`. Rinciannya di `docs/schema.md`.
  Contoh nyata: `[{"name":"get_mk_diampu_dosen","args":{"matkul":"Basis
  Data"},"ok":true,"latency_ms":909}]`.
- **`messages.meta.attachments`**: lampiran yang tampil di bawah jawaban, utuh
  (`[{title, source, items}]`); `null` bila tidak ada. Disimpan lengkap karena
  data SADS berubah per semester: memanggil ulang tool tidak mengulang daftar
  yang dilihat mahasiswa saat ia menilai 👎.
- **LangSmith**: setiap giliran LLM di loop menjadi run `generate_answer`
  tersendiri di bawah trace giliran. Panggilan tool **tidak** punya span sendiri;
  jejaknya ada di `meta.tool_calls`.
- **Token**: `usage` dijumlahkan lintas giliran (§2). Pertanyaan ber-tool
  memakai sekitar 7–8 ribu token input dengan `cx/gpt-6-luna`, karena skema tool,
  aturan T1–T6, dan hasil tool. `get_daftar_dosen` tanpa saringan membawa sekitar
  220 nama (~13 ribu token input untuk dua giliran). Dengan lampiran, keluarannya
  tidak lagi memuat daftar itu: sebelumnya ~3,8 ribu token keluaran untuk
  "sebutkan semua dosen".
- **Latensi**: minimal dua round-trip LLM ditambah SADS (sekitar 0,2–1,3 dtk per
  panggilan dalam uji live).

---

## 14. Konfigurasi (env)

```text
TOOLS_ENABLED=false            # sakelar utama; bawaan kode mati
TOOLS_MAX_ROUNDS=2             # batas putaran loop agentik (1–5)
SADS_BASE_URL=https://sads.instiki.ac.id
SADS_API_SECRET=<secret>       # header `secret`
SADS_TIMEOUT_SECONDS=10
```

Polanya mengikuti `RERANK_*`: fitur di belakang sakelar, bawaan mati, nilai di
`config.py` dan `.env.example`.

---

## 15. Testing

`tests/unit/test_tools.py`, tanpa jaringan (LLM dan SADS dipalsukan):

- **Registry**: tool terdaftar, skema OpenAI, kelayakan (Indonesia, Inggris,
  di luar topik), seluruh registry di-bind.
- **Validasi argumen**: karakter kontrol dibuang, argumen asing dibuang, argumen
  wajib kosong ditolak, argumen di `meta` dipotong.
- **Handler SADS**: normalisasi, dedup, jumlah; `SadsClient` mengirim header
  `secret` (bukan `Authorization`) dan query `matkul` lewat `httpx.MockTransport`;
  status galat menjadi `HTTPStatusError`.
- **Loop**: tool lalu jawaban; hanya giliran final yang di-stream; argumen
  dihalusinasi tetap divalidasi; tool gagal tidak menjatuhkan giliran; batas
  putaran; eksekusi **bersamaan** (dua handler yang saling menunggu); batas 4 per
  giliran; `invalid_tool_calls`; giliran final kosong; `[Error]` gateway;
  urutan stage; satu config per giliran.
- **Integrasi pipeline**: konteks kosong + tool-eligible → `answer`; tanpa
  registry → `refusal`; tool tak tersedia → penolakan tanpa LLM; konteks kuat
  tetap dijawab; di luar topik tidak eligible; pertanyaan lanjutan lewat rewrite
  (dan pertanyaannya di dalam tag); jawaban parsial tool tidak dibuang.
- **Meta**: `tool_calls` masuk `messages.meta`; `null` tanpa tool.
- **Saringan `get_daftar_dosen`** (`TestSaringDaftarDosen`): gelar dihitung
  handler, "Dr." tidak mencocokkan "Drs.", penulisan gelar disamakan, sinonim
  "doktor", `nama` hanya ke nama inti, gabungan nama + gelar, saringan kosong
  tetap `ok`, SADS kosong tetap gagal, argumen opsional `null`.
- **Lampiran** (`TestLampiran`): sisipan `DAFTAR_DITAMPILKAN` sesuai aturan T5,
  loop mengumpulkan lampiran, pipeline meneruskan hanya yang sumbernya dikutip,
  penolakan tanpa lampiran, `ChatResponse` dan `messages.meta` membawanya.
- **`get_mk_diampu_dosen`** (`TestHandlerSads`): hasil kosong tetap `ok` tanpa
  `DATA_TIDAK_TERSEDIA`, varian huruf ganda ditanyakan dan digabung per dosen,
  kata kunci tanpa huruf ganda satu panggilan, `description` mengarah ke nama
  Inggris dan `get_mk_dosen`.
- **`get_mk_dosen`** (`TestMkDosen`): SADS dipanggil tanpa `matkul`, satu dosen
  dirinci + dihitung + berlampiran, sapaan dan gelar dibuang, nama umum hanya
  nama-nama, beberapa dosen tanpa lampiran, 0 hasil tetap `ok`, SADS kosong
  gagal, pemicu Indonesia/Inggris.

Yang **tidak** diuji otomatis: ketahanan model terhadap instruksi yang disisipkan
di hasil tool. Itu perilaku model, dan pertahanannya ada di aturan T2 + aturan 5.

---

## 16. Checklist menambah endpoint

1. Tulis handler di `app/rag/tools/sads.py`, atau modul layanan lain: panggil
   endpoint, normalisasi ke teks ringkas, sertakan jumlah bila datanya berupa
   daftar, lalu kembalikan `ToolResult`. Daftar yang mungkin panjang dan tidak
   perlu disaring model sebaiknya juga dikirim sebagai `Lampiran` (§10a);
   saringan yang mungkin ditanyakan ("bergelar Dr.") jadikan argumen, bukan
   tugas model.
2. Deklarasikan `ToolSpec`: `name`, `description` (sempit dan jelas),
   `parameters` (JSON Schema), `triggers` (Indonesia **dan** Inggris), dan
   `citation_label`.
3. Daftarkan di `tool_specs` (dipungut `build_registry`).
4. Tambah test handler dan skema (§15).
5. Bila perlu env baru (base/secret layanan lain), tambahkan di `config.py` dan
   `.env.example`.

Loop, streaming, sitasi, dan gerbang kelayakan **tidak** perlu disentuh. Begitu
registry memuat banyak layanan, kelompokkan tool per layanan dan bind hanya
kelompok yang cocok, supaya skema tool tidak membebani setiap prompt.

---

## 17. Rencana bertahap

Semua fase sudah dijalankan (2026-10-07):

1. ✅ **Fondasi**: `ToolSpec`, registry, klien HTTP, loop, di belakang `TOOLS_ENABLED`.
2. ✅ **Streaming**: giliran tool tidak memancarkan token; penyaring `[Error]`
   dan penahan penanda dipakai ulang.
3. ✅ **Integrasi gerbang**: `tool_eligible` di `validate_context`, kartu sumber
   sintetis di `generate`.
4. ✅ **Observability**: `messages.meta.tool_calls` dan `usage`.
5. ✅ **Nyalakan**: `TOOLS_ENABLED=true` di `.env` lokal, terverifikasi live.
   Untuk produksi, set variabelnya di env server.

---

## 18. Riwayat tinjauan

### Tinjauan pertama (2026-10-07)

| Cacat | Akibat | Perbaikan |
|---|---|---|
| Invarian FR-3 bocor | tool tak tersedia + konteks lemah memanggil LLM dengan KONTEKS kosong, tercatat `refusal_source = llm` | `generate` menolak tanpa LLM |
| Pemicu tidak simetris | "ada berapa dosen?" hanya mem-bind tool yang mewajibkan `matkul` | seluruh tool di-bind |
| Uji coba admin tanpa tool | admin melihat `refusal`, mahasiswa menerima jawaban | `test_query` meneruskan registry |
| Argumen mentah ke `meta` | string raksasa menggelembungkan kolom | dipotong 200 karakter |
| Field `ToolSpec.unit` mati | menyiratkan scoping per-unit yang tidak ada | dibuang |

### Tinjauan kedua (2026-10-07)

| Cacat | Akibat | Perbaikan |
|---|---|---|
| **Jumlah dari model salah** | "Terdapat 223 dosen" dari 220 nama, lengkap dengan kartu sumber yang membuatnya tampak sah | jumlah dihitung handler; live: "Ada 220 dosen" |
| Jawaban parsial tool dibuang | `[Data akademik SADS]` tak terbaca sebagai sitasi, jadi "… tidak ditemukan" menjadikan seluruh jawaban `refusal` | sumber tak berhalaman diteruskan ke `is_not_found` / `is_off_topic` |
| Pertanyaan lanjutan tidak memicu tool | kelayakan hanya dari teks asli, loop tanpa riwayat | kelayakan juga dari hasil rewrite; loop menerima versi mandirinya |
| Pertanyaan Inggris pendek ditolak | "who teaches Web Programming?" tidak di-rewrite, tidak ada pemicu Inggris | pemicu `lecturer`, `teach` |
| `invalid_tool_calls` diabaikan | argumen bukan JSON sah membuat giliran dikira jawaban final yang kosong | dibalas pesan galat, loop berlanjut |
| Giliran final kosong | tampil sebagai `answer` kosong | diganti `[TIDAK_DITEMUKAN]` |
| Klaim "paralel" dan "batas per giliran" tidak dipenuhi kode | tool berjalan berurutan dan tanpa batas | `asyncio.gather`, maksimal 4 per giliran |
| Satu `run_id` dipakai semua giliran loop | run berikutnya bentrok di LangSmith; giliran jawaban berisiko hilang dari trace (disimpulkan dari kode, tidak diamati karena tracing mati saat uji) | config baru per giliran |
| Status SSE tertahan "Mencari dokumen…" | di jalur tool tidak ada `menyusun jawaban` sebelum loop | urutan stage §9 |
| Test yang diklaim dokumen ini tidak ada | header `secret`, paralel, `[Error]` di loop | ditambahkan; `SadsClient` menerima `transport` |
| Dokumen ini usang | kartu "per <waktu>", fallback kontak saat batas putaran, `ToolSpec.unit`, span LangSmith per tool | ditulis ulang |

### Terverifikasi live (2026-10-07)

`POST /api/chat` dengan retriever (dev DB), JEV, rewrite, LLM, dan SADS asli.
Hanya pencatat percakapan yang diganti, supaya statistik tidak tercemar.

| Pertanyaan | JEV | Retrieval | Hasil |
|---|---|---|---|
| siapa dosen pengampu mata kuliah Web Programming? | academic 0,97 | top 0,83 (lolos FR-3) | `answer`, `get_mk_diampu_dosen("Web Programming")` |
| kalau Basis Data? (lanjutan) | academic 0,97 | top 0,85 | rewrite → "Siapa dosen pengampu mata kuliah Basis Data?" → `answer` |
| ada berapa dosen di INSTIKI? | academic 0,96 | top 0,88 | `get_daftar_dosen` → "Ada 220 dosen" |
| who teaches Web Programming? | academic 0,49 | top 0,79 | `answer` dalam bahasa Indonesia, setelah pemicu Inggris ditambahkan (sebelumnya `refusal`) |

Juga terverifikasi: SSE `/api/chat/stream` tanpa JSON tool-call di token, dan
jalur batas putaran (riwayat `role:"tool"` tanpa `tools`) diterima gateway.

Retrieval selalu lolos FR-3 untuk pertanyaan dosen, karena skor vektor e5
memang tinggi untuk hampir semua pertanyaan (`flow.md` §6). Model tetap memilih
tool walau diberi chunk PDF yang tidak relevan.

### Saringan dan lampiran (2026-10-07)

Uji browser pada widget dan Uji coba menemukan dua masalah pada daftar 220
dosen. Pertama, hitungan dan saringan yang dikerjakan model salah walau tampak
bersumber: "bergelar Dr." dijawab 38 lalu 39 dari 25 nama, dan "nama ada Wayan"
dijawab 14 dari 15 nama (T46). Kedua, daftar lengkap yang disalin model memakan
3,8 ribu token keluaran, 27–70 detik menulis, dan gelembung setinggi 17 layar.
Perbaikannya: argumen `nama` dan `gelar` (§7) dan lampiran (§10a).

| Pertanyaan (live, LLM + SADS asli) | Sebelum | Sesudah |
|---|---|---|
| berapa jumlah dosen INSTIKI yang sudah bergelar Dr.? | "38", "39" | 25, lampiran 25 nama (3/3 di uji ulang) |
| siapa saja dosen yang namanya ada Wayan? | 14 dari 15 nama | 15, lampiran 15 nama |
| sebutkan daftar nama semua dosen di INSTIKI | 3,8 ribu token keluaran, 101 dtk | 62 token keluaran, ~12–23 dtk, lampiran 220 nama per 10 |
| apakah ada dosen bergelar profesor di INSTIKI? | (belum diuji) | "Tidak ada … 0 orang dari 220", bersumber, tanpa lampiran |
| apakah Totok Suryawan dosen di INSTIKI? | (belum diuji) | "Ya, Dr. I Gede Totok Suryawan … mengajar di INSTIKI" |
| siapa dosen pengampu mata kuliah Web Programming? | 23 nama | 23 nama (tool tanpa lampiran; lihat di bawah) |

Regresi yang tertangkap saat uji live: aturan T5 versi pertama ("jangan
menyalin daftar, salin jumlahnya, jawab ringkas") berlaku untuk semua hasil
alat, sehingga pertanyaan Web Programming, yang tidak berlampiran, dijawab
"berjumlah 23 orang" tanpa nama. Ablasi langsung ke loop (4 percobaan per
varian): T5 umum 3/4 menyebut nama dan selalu dibuka dengan jumlah; tanpa T5
4/4; T5 yang hanya mengesahkan penanda 4/4. Versi terakhir itu juga 9/9 ringkas
dan benar pada pertanyaan berlampiran, jadi petunjuknya dipindah ke
`CATATAN_LAMPIRAN`, yang hanya menyertai hasil berlampiran.

### Mata kuliah: nama Inggris dan per dosen (2026-10-07)

Uji browser `get_mk_diampu_dosen` di widget dan Uji coba, dicocokkan ke SADS
mentah, menemukan T47 (nama mata kuliah Indonesia vs Inggris) dan T48 (mata
kuliah per dosen tidak didukung). Perbaikannya di §7, aturan T6, dan
`CATATAN_LAMPIRAN` (§10a).

| Pertanyaan (topik Prodi, LLM + SADS asli) | Sebelum | Sesudah |
|---|---|---|
| Siapa dosen pengampu Jaringan Komputer? | 8/8, "Lanjut" tersaring | (tidak diubah) |
| Siapa dosen yang mengajar Kecerdasan Buatan? | penolakan + kontak FO (`matkul` "Kecerdasan Buatan" → 0) | "Kecerdasan Buatan" → 0 → "Artificial Intelligence" → 35; nama disebut 35/35 ×3, 34/35 ×2, 4 nama "antara lain" ×1 |
| Mata kuliah apa saja yang diajar oleh Ahmad Asroni? | `matkul="Ahmad Asroni"` → "belum dapat diambil" / penolakan | `get_mk_dosen` → "25 mata kuliah", lampiran 25 |
| Pak Wayan ngajar mata kuliah apa aja? | (belum diuji) | 15 nama, meminta nama lengkap |
| Apakah Pak Totok mengajar Basis Data? | (belum diuji) | "Ya", lampiran 17 |
| Siapa dosen pengampu Basis Data dan Kalkulus? | 1 panggilan gabungan → 0 → penolakan | dua panggilan + "Calculus"; Basis Data 10/10, Kalkulus "tidak ditemukan" |
| Siapa saja dosen yang bergelar Dr.? (regresi `CATATAN_LAMPIRAN`) | 25, ringkas | 25, ringkas |

### Masih terbuka (sengaja tidak diubah tanpa keputusan)

1. **`eval/run_generation.py` tidak memakai tool.** RAGAS melihat `refusal` untuk
   pertanyaan yang di produksi dijawab tool. Tidak diwire supaya metodologi
   evaluasi yang sedang berjalan tidak berubah diam-diam.
2. **Penolakan jalur tool masuk AD-4.** Padahal itu bukan celah dokumen: menambah
   PDF tidak menyelesaikannya, dan dengan alasan yang sama `rejected` sengaja
   tidak masuk AD-4. Penolakan ini dapat dibedakan lewat `meta.tool_calls`.
3. **Jawaban tool dapat membawa `ThresholdDecision` REFUSE**, yaitu bila konteks
   retrieval lemah. Akibatnya `messages.top_score` rendah untuk jawaban yang
   benar, dan `applog` mencatat `decision=refuse` di `validate_context` untuk
   giliran yang berakhir `answer`.
4. **Kriteria `academic` JEV tidak menyebut dosen atau mata kuliah**, sementara
   `out_of_scope` menyebut "general coding or technology help". Keempat uji live
   lolos, tetapi "who teaches Web Programming?" hanya mendapat 0,49. Mengubah
   kriterianya berarti mengkalibrasi ulang JEV (`jev.md`).
5. **Hasil SADS tidak di-cache.** Setiap pertanyaan ber-tool memanggil SADS, dan
   `get_daftar_dosen` membawa sekitar 220 nama ke prompt. Datanya berubah per
   semester, jadi cache singkat (TTL) aman bila beban atau biaya menjadi masalah.
6. **Asumsi giliran tool tanpa konten.** Model yang mengirim konten *dan*
   `tool_calls` dalam satu giliran akan membuat konten itu ikut mengalir sebelum
   jawaban final.
7. ~~`get_mk_diampu_dosen` tanpa hasil dibalas `DATA_TIDAK_TERSEDIA`.~~ Selesai
   2026-10-07 (T43/T47, §7). Sisa: daftar panjang dari tool ini tetap disalin
   model dan kadang kehilangan nama (§10a, Batasan).
