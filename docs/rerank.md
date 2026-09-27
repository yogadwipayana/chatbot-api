# Reranker

Langkah opsional yang memperbaiki urutan potongan dokumen sebelum dikirim ke
LLM. Diatur segmen `Reranker` di `.env`. Bawaannya **mati**
(`RERANK_PROVIDER=none`): selama itu seluruh variabel `RERANK_*` tidak
berpengaruh apa pun.

## Masalah yang diselesaikan

Retrieval berjalan di dua jalur -- pencarian makna (vektor) dan pencarian kata
(fulltext) -- lalu hasilnya digabung dengan RRF. RRF hanya menggabungkan
**peringkat** kedua jalur; ia tidak pernah membaca pertanyaan dan potongan
dokumen bersamaan. Potongan yang kebetulan memuat kata kunci yang sama tetapi
tidak benar-benar menjawab bisa ikut masuk lima besar.

Reranker (cross-encoder) menilai setiap pasangan *(pertanyaan, potongan)*
secara utuh, lalu mengurutkan ulang:

```
tanpa reranker:  vektor + fulltext → RRF → 5 teratas → LLM
dengan reranker: vektor + fulltext → RRF → 20 teratas → reranker → 5 teratas → LLM
```

Dua sifat yang perlu diketahui:

- **Kegagalan tidak menggagalkan jawaban.** Bila reranker galat atau melewati
  `RERANK_TIMEOUT_SECONDS`, urutan RRF dipakai apa adanya dan galatnya dicatat.
- **Skornya mutlak (0–1)**, tidak seperti skor RRF. Karena itu ia boleh menjadi
  dasar penolakan (FR-3) lewat `RERANK_THRESHOLD`.

## Variabel

| Variabel | Fungsi |
|---|---|
| `RERANK_PROVIDER` | `none` (mati), `api` (endpoint `/rerank` gaya Cohere/Jina), atau `local` (cross-encoder di server sendiri) |
| `RERANK_MODEL` | Nama model, mis. `BAAI/bge-reranker-v2-m3`. Wajib bila provider bukan `none`. |
| `RERANK_BASE_URL`, `RERANK_API_KEY` | Endpoint dan kuncinya (mode `api`); kosong = `BASE_URL` / `API_KEY` |
| `RERANK_CANDIDATES` | Berapa hasil RRF teratas yang dinilai ulang (bawaan 20). Yang masuk konteks LLM tetap `RETRIEVAL_TOP_N`. |
| `RERANK_THRESHOLD` | Opsional, 0–1. Terisi = penolakan FR-3 memakai skor reranker, bukan `VECTOR_THRESHOLD`/`LEXICAL_THRESHOLD`. Kosong = ambang lama. |
| `RERANK_TIMEOUT_SECONDS` | Batas waktu satu penilaian (bawaan 10 detik) |

Semuanya dibaca dari `.env` saja -- tidak dapat diubah dari halaman
Konfigurasi dashboard -- jadi setiap perubahan butuh restart API. Setelan yang
setengah terisi (provider tanpa model, mode `api` tanpa URL) membuat API gagal
start dengan pesan yang jelas, bukan gagal saat mahasiswa bertanya.

## Menyalakan

**Lewat endpoint (`api`)** -- butuh layanan dengan `POST {url}/rerank` yang
menerima `{model, query, documents, top_n}` dan menjawab
`{results: [{index, relevance_score}]}` (bentuk Cohere, Jina, Voyage). Belum
diperiksa apakah gateway `BASE_URL` proyek ini menyediakannya.

```bash
RERANK_PROVIDER=api
RERANK_MODEL=<nama model di penyedia>
RERANK_BASE_URL=           # kosong = BASE_URL
RERANK_API_KEY=            # kosong = API_KEY
```

**Di server sendiri (`local`)** -- butuh sentence-transformers, serta CPU dan
RAM untuk modelnya:

```bash
cd api && uv sync --extra local
# .env
RERANK_PROVIDER=local
RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

Lalu restart API. Mulailah dengan `RERANK_THRESHOLD` kosong: reranker hanya
memperbaiki urutan, sementara keputusan menolak tetap memakai ambang lama.

## Ambang `RERANK_THRESHOLD`

Belum ada skrip kalibrasi untuk skor reranker. Skornya tercatat di setiap
jawaban -- `messages.meta.top_rerank_score`, dan "skor rerank" di rincian
giliran halaman Log -- jadi ambangnya ditetapkan dari data:

1. Nyalakan reranker dengan `RERANK_THRESHOLD` kosong, biarkan berjalan
   beberapa hari.
2. Bandingkan `top_rerank_score` pertanyaan yang seharusnya terjawab dengan yang
   seharusnya ditolak (mis. yang dinilai 👎 atau masuk daftar pertanyaan tak
   terjawab).
3. Isi ambang di antara keduanya, restart, lalu buktikan di halaman Uji coba.

Ambang yang terlalu tinggi membuat chatbot menolak pertanyaan yang sebenarnya
ada jawabannya; terlalu rendah membuatnya menjawab dari dokumen yang tidak
relevan.

## Kapan perlu dinyalakan

Bila di halaman **Uji coba** dashboard terlihat potongan yang relevan ada di
daftar, tetapi kalah peringkat dari potongan yang kurang tepat -- sehingga
jawabannya mengutip dokumen yang salah padahal dokumen yang benar sudah ada.

Konsekuensinya:

- satu panggilan tambahan per pertanyaan, jadi jawaban sedikit lebih lambat;
- mode `local` memakai CPU/RAM server; mode `api` menambah biaya per panggilan
  di penyedia.

## Berkas terkait

| Berkas | Isi |
|---|---|
| `app/rag/reranker.py` | `ApiReranker`, `LocalReranker`, `build_reranker`, `rerank_documents` |
| `app/rag/retriever.py` | Memotong hasil RRF ke `RERANK_CANDIDATES` sebelum reranker |
| `app/rag/threshold.py` | Keputusan menolak memakai skor reranker bila `RERANK_THRESHOLD` diisi |
| `app/config.py` | Variabel `RERANK_*` dan validasinya |
