# Reranker

Langkah opsional yang memperbaiki urutan potongan dokumen sebelum dikirim ke
LLM. Diatur segmen `Reranker` di `.env`. Bawaannya **mati**
(`RERANK_ENABLED=false`, seperti `JEV_ENABLED`): selama itu variabel
`RERANK_*` lainnya tidak berpengaruh apa pun, jadi boleh tetap terisi.

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
| `RERANK_ENABLED` | Sakelar. `false` (bawaan) = urutan RRF langsung dipakai. |
| `RERANK_PROVIDER` | Bentuk endpoint, bukan model: `tei` (bawaan), `api`, atau `local`. Lihat tabel di bawah. |
| `RERANK_BASE_URL` | Alamat server reranker, mis. `http://localhost:8081`; `/rerank` ditambahkan otomatis. Wajib kecuali `local`. **Tidak** jatuh ke `BASE_URL`, karena gateway tidak punya `/rerank`. |
| `RERANK_API_KEY` | Dikirim sebagai `Authorization: Bearer`. Kosong = tanpa header. **Tidak** jatuh ke `API_KEY`, supaya kunci gateway tidak terkirim ke server lain. |
| `RERANK_MODEL` | Nama model, mis. `Alibaba-NLP/gte-multilingual-reranker-base`. Wajib saat menyala. TEI mengabaikannya (satu server satu model), jadi di sana hanya untuk log. |
| `RERANK_CANDIDATES` | Berapa hasil RRF teratas yang dinilai ulang (bawaan 20). Yang masuk konteks LLM tetap `RETRIEVAL_TOP_N`. |
| `RERANK_THRESHOLD` | Opsional, 0–1. Terisi = penolakan FR-3 memakai skor reranker, bukan `VECTOR_THRESHOLD`/`LEXICAL_THRESHOLD`. Kosong = ambang lama. |
| `RERANK_TIMEOUT_SECONDS` | Batas waktu per permintaan ke server reranker (bawaan 10 detik) |

| `RERANK_PROVIDER` | Permintaan → jawaban | Contoh server |
|---|---|---|
| `tei` | `{query, texts, raw_scores, truncate}` → `[{index, score}]`, maksimal 32 teks per permintaan (dipecah otomatis) | Text Embeddings Inference, mis. container di `/rerank` server |
| `api` | `{model, query, documents, top_n}` → `{results: [{index, relevance_score}]}` | Cohere, Jina, Voyage, Infinity |
| `local` | Cross-encoder sentence-transformers di proses API; butuh `uv sync --extra local` | -- |

Semuanya dibaca dari `.env` saja -- tidak dapat diubah dari halaman
Konfigurasi dashboard -- jadi setiap perubahan butuh restart API. Setelan yang
setengah terisi (menyala tanpa model, `tei`/`api` tanpa URL, URL tanpa
`http://`) membuat API gagal start dengan pesan yang jelas, bukan gagal saat
mahasiswa bertanya. `.env` lama yang masih berisi `RERANK_PROVIDER=none` juga
ditolak dengan pesan yang menunjuk ke `RERANK_ENABLED`.

## Menyalakan

Contoh dengan container gte di `/rerank` server (panduan container:
`/rerank/README.md` di server):

```bash
RERANK_ENABLED=true
RERANK_PROVIDER=tei
RERANK_BASE_URL=http://localhost:8081   # dari laptop: http://100.111.178.48:8081
RERANK_API_KEY=<GTE_API_KEY di /rerank/.env>
RERANK_MODEL=Alibaba-NLP/gte-multilingual-reranker-base
RERANK_THRESHOLD=
```

Lalu restart API. Di server, container API harus dibuat ulang supaya `.env`
terbaca: `cd /chatbot && sudo docker compose up -d api`. Mulailah dengan
`RERANK_THRESHOLD` kosong: reranker hanya memperbaiki urutan, sementara
keputusan menolak tetap memakai ambang lama.

Mematikan cukup `RERANK_ENABLED=false` lalu restart; variabel lain boleh
dibiarkan.

## Mengganti model

Cukup tiga variabel, lalu restart API. Misalnya gte → bge:

```bash
RERANK_BASE_URL=http://localhost:8082
RERANK_API_KEY=<BGE_API_KEY di /rerank/.env>
RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

`RERANK_PROVIDER` hanya diganti bila jenis servernya berganti (mis. dari TEI ke
layanan gaya Cohere). Kosongkan `RERANK_THRESHOLD` setiap ganti model, lalu
kalibrasi ulang: sebaran skor tiap model berbeda.

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
- `tei` dan `local` memakai CPU/RAM server (CPU server proyek ini tanpa AVX,
  jadi ukur dulu latensinya); `api` ke penyedia berbayar menambah biaya per
  panggilan.

## Berkas terkait

| Berkas | Isi |
|---|---|
| `app/rag/reranker.py` | `TeiReranker`, `ApiReranker`, `LocalReranker`, `build_reranker`, `rerank_documents` |
| `app/rag/retriever.py` | Memotong hasil RRF ke `RERANK_CANDIDATES` sebelum reranker |
| `app/rag/threshold.py` | Keputusan menolak memakai skor reranker bila `RERANK_THRESHOLD` diisi |
| `app/config.py` | Variabel `RERANK_*` dan validasinya |
