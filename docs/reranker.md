# Reranker: status dan temuan yang menunggu

Catatan kerja untuk reranker, yang **belum dipakai**. Semua informasi terkait reranker ditulis di sini, bukan di `browseract.md`. Panduan teknisnya (variabel, cara menyalakan, kalibrasi ambang) ada di `api/docs/rerank.md`.

## Status (per 2026-09-29)

- **Ditunda.** gte dan bge sama-sama terlalu lambat di CPU server, dan keduanya menurunkan Pasal 10 sehingga merusak jawaban larangan kode etik yang sudah benar (lihat *Hasil uji gte* dan *Hasil uji bge di browser*). Setelah uji itu (2026-09-29), user memilih "jalan tengah". Kodenya tetap ada dalam keadaan mati. Segmen reranker di `api/.env` dan `.env.example` tinggal `RERANK_ENABLED=false`; URL, kunci, dan model bge dihapus, dan variabel lain memakai nilai bawaan `config.py`. Container gte/bge beserta volume `gte-data` dan `bge-data` sudah dihapus dari server (disk kembali 4,8 GB bebas), jadi menyalakan lagi berarti mengunduh ulang model. Jangan tawarkan reranker sebagai opsi perbaikan di `browseract.md` sampai status ini berubah.
- **Kode siap (2026-09-29).** Sakelarnya `RERANK_ENABLED` (seperti `JEV_ENABLED`), bawaannya `false`. Adapter TEI ada (`RERANK_PROVIDER=tei`, bawaan). Ganti model cukup lewat `RERANK_BASE_URL`, `RERANK_API_KEY`, dan `RERANK_MODEL`, lalu restart API. Panduan lengkap: `api/docs/rerank.md`.
- **Container di server (2026-09-29).** `/rerank/docker-compose.yml` di `embed@100.111.178.48` berisi TEI `gte-multilingual-reranker-base` di :8081 dan `bge-reranker-v2-m3` di :8082. Project ini berdiri sendiri (`rerank-network`). Panduannya di `/rerank/README.md`, dan `/rerank/.env` berisi kunci `GTE_API_KEY`/`BGE_API_KEY`. Folder ini sengaja disimpan (berisi setelan anti-OOM), tapi container dan volumenya sudah dihapus (lihat status di atas).
- **Insiden OOM (2026-09-29).** Dengan setelan bawaan TEI, warmup bge (`MAX_BATCH_TOKENS` 16384 × konteks 8192) butuh lebih dari 9 GB. Host terkena *global OOM* 16 kali dan bge restart 33 kali sebelum dihentikan. pg, API, dan e5 selamat. Perbaikannya ada di compose: `MAX_BATCH_TOKENS=2048` dan `mem_limit` (gte 4g, bge 6g). Memuat bobot butuh jauh lebih banyak dari ukuran filenya: gte (584 MB) mati di 3 GB, bge (2,3 GB) mati di 4 GB.
- **Latensi bge (2026-09-29):** 20 potongan × 900 karakter butuh **±19 detik**, diukur dari laptop lewat Tailscale ke `TeiReranker`. Itu jauh melewati `RERANK_TIMEOUT_SECONDS=10`, jadi bge **tidak layak** di CPU ini. Bentuk jawaban TEI (daftar `[{index, score}]`) dan kuncinya sudah terbukti terbaca oleh adapter.

## Hasil uji gte (2026-09-29)

Diukur dari laptop ke `TeiReranker`. Kandidatnya 20 hasil RRF asli dari database (`neighbors=0`), yaitu kandidat yang akan dinilai reranker di chatbot.

- **Latensi: 9,3–11,7 detik** per pertanyaan. Sintetis 9,7 / 10,4 / 11,7 detik; kandidat asli (±11.500 karakter) 9,3–9,8 detik. Hasil ini tepat di batas 10 detik, jadi sebagian pertanyaan akan timeout dan kembali ke urutan RRF. gte hanya sekitar 2× lebih cepat dari bge, bukan 3,5× seperti perkiraan.
- **T9 "Apa saja jenis beasiswa yang tersedia?" (Kemahasiswaan): membaik sebagian.** h27 "6.1 Gambaran Umum" naik dari RRF#11 ke **#4**, tapi h4 tetap di luar 8 besar. Skornya rapat (0,52 / 0,44 / 0,42 / 0,417 / 0,415 …), jadi gte kurang tegas membedakan.
- **Larangan kode etik (BAAK): MUNDUR.** Pasal 10 h8 "Mahasiswa INSTIKI dilarang: …" turun dari RRF#3 ke **#10** (skor 0,51), jadi keluar dari 5 besar. Yang naik ke atas adalah Pasal 12 (sanksi, 0,79) dan Pasal 5/7/4/6. Pertanyaan ini sekarang sudah terjawab tanpa reranker, jadi gte akan merusaknya.
- **T3 (pemisahan skor): menjanjikan, tapi baru 3 contoh.** Skor tertinggi untuk pertanyaan yang tak terjawab ("Berapa biaya parkir motor di kampus?", Kemahasiswaan) 0,22, sedangkan yang terjawab 0,52 dan 0,79. Skor vektor e5 untuk ketiganya tetap sama-sama 0,83–0,89.
## Hasil uji bge di browser (2026-09-29, sesi browseract 14)

User mengisi `RERANK_API_KEY` dan `RERANK_ENABLED=true` (bge, `:8082`). Uji lewat widget dan Uji coba memakai sesi baru `uji-rr-*` (sudah dihapus), ditambah skrip yang sama seperti uji gte.

- **Dengan `RERANK_TIMEOUT_SECONDS=10` (setelan user):** reranker selalu gagal dengan `httpx.ReadTimeout` lalu kembali ke urutan RRF. `retrieve` bertambah sekitar 10 detik tanpa manfaat apa pun.
- **Dengan timeout 30 (sementara, lalu dikembalikan ke 10):** rerank butuh 16–19 detik. Selama rerank, bge memakai ±990% CPU (ke-10 core penuh), jadi lambatnya karena komputasi, bukan karena satu thread.
- **Beasiswa (T9): tidak membaik.** h27 naik dari RRF#11 ke #8 dan h4 dari #15 ke #9, keduanya tetap di luar 5 besar. Skor rerank tertinggi 0,44. Widget menjawab dua jenis beasiswa (hal. 20 dan 32), sama seperti tanpa reranker.
- **Larangan kode etik: MUNDUR, terbukti di browser.** Pasal 10 turun dari RRF#3 ke #10 (skor 0,33). Jawaban widget disusun dari Pasal 4–7 dan menyatakan "kutipan belum memuat seluruh larangan", padahal di sesi 11 Pasal 10 lengkap dikutip. Skor rerank tertinggi 0,935 (Pasal 12).
- **T3: pemisahan sangat tegas.** "Biaya parkir motor" mendapat skor tertinggi 0,001, sedangkan yang terjawab 0,44 dan 0,935. Hasil ini jauh lebih tegas dari gte (0,22 vs 0,52).
- **Halaman Uji coba tidak menampilkan skor rerank.** Respons `/api/admin/test-query` tidak memuat `rerank_score` per potongan maupun `top_rerank_score`. Skor rerank hanya muncul di rincian giliran halaman Log ("skor rerank: 0,935"), jadi langkah 2 di bawah perlu memakai halaman Log.
- **Kesimpulan:** bge tidak layak sebagai pengurut ulang, karena lambat, merusak Pasal 10, dan tidak membantu T9. Sebagai sinyal penolakan (T3) bge menjanjikan, tapi 16–19 detik terlalu mahal.

## Kesimpulan uji model

- **gte tidak dinyalakan.** Pilihan yang tersisa: model yang lebih kecil (mis. `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, ±4× lebih ringan, mutu untuk bahasa Indonesia belum diketahui), versi ONNX int8 (TEI memakai backend ORT kalau `model.onnx` ada), atau server dengan AVX/GPU. Kalau tidak dipakai, container gte sebaiknya dimatikan karena memakan ±3,4 GB RAM di host produksi.
- **Kondisi server:** 10 vCPU QEMU **tanpa AVX/AVX2** (hanya SSE4.2), RAM 9,7 GB, sisa disk 4,8 GB (84%) setelah volume model dihapus.
- **Server `/chatbot` belum memakai kode baru.** `.env`-nya masih `RERANK_PROVIDER=none`. Saat kode baru di-deploy, `.env` server ikut diperbarui. Kalau tidak, API gagal start dengan pesan yang menunjuk ke `RERANK_ENABLED`.
- Gateway `BASE_URL` **tidak punya** endpoint `/rerank` (dicek 2026-09-28), jadi `RERANK_BASE_URL` wajib diisi dan tidak jatuh ke `BASE_URL`.

## Temuan yang menunggu reranker

### T3 (penting): skor e5 tidak bisa memisahkan pertanyaan yang relevan dan yang tidak
- Dipindah dari `browseract.md` §5 pada 2026-09-29. Ditemukan 2026-09-28 (sesi 2).
- Skor e5 untuk pertanyaan yang terjawab dan yang tidak terjawab sama-sama 0,80–0,86. Akibatnya ambang hanya menolak pertanyaan ke unit yang belum punya dokumen (mis. jadwal wisuda di topik Akademik). Di unit yang punya dokumen, keputusan menolak diserahkan ke LLM penjawab.
- Perbaikan dengan reranker: skornya mutlak (0–1), jadi `RERANK_THRESHOLD` bisa menggantikan ambang vektor/leksikal sebagai dasar penolakan.

### T5/T9 (sedang): peringkat pencarian (bagian reranker saja)
- T9 selesai tanpa reranker (2026-10-09 sesi 43): bab Berprestasi dan Talenta sebenarnya sudah ikut terambil, tetapi model hanya mengikuti daftar empat jenis di TRANSKRIP. Perbaikannya daftar bab dokumen di konteks (`RETRIEVAL_OUTLINE`, `app/rag/outline.py`), bukan peringkat. T5 (Pasal 10 terpotong) sudah selesai tanpa reranker, lewat `RETRIEVAL_NEIGHBORS=5` (2026-09-29 sesi 11).
- Peran reranker: mengurutkan ulang 20 kandidat RRF supaya potongan yang benar-benar menjawab naik ke 5 teratas.
- **Kasus terkuat: "Apa saja jenis beasiswa yang tersedia?" (Kemahasiswaan).** Ditolak LLM 4/4 kali, baik di nilai NEIGHBORS 2 maupun 5. Potongan yang menjawab ada di peringkat RRF **11** (h27 "6.1 Gambaran Umum", Beasiswa Prestasi Internal) dan **15** (h4, keputusan rektor), jadi masih di dalam 20 kandidat yang dinilai reranker. Lima besar saat ini diisi h20/h32/h33 (profil program, Talenta Unggul, evaluasi). Dengan kalimat "…bagi mahasiswa INSTIKI?", keduanya naik ke 5 besar dan pertanyaannya terjawab sebagian.
- Sudah tidak relevan: "Kapan pendaftaran sertifikasi dibuka?" (UPS) sekarang terjawab 5/6 kali (pengumuman di awal, tengah, dan akhir semester) tanpa reranker.

## Langkah setelah model siap

1. Pilih model yang latensinya jauh di bawah 10 detik untuk 20 kandidat asli, dan yang tidak menurunkan Pasal 10 dari 3 besar. gte dan bge gagal (*Hasil uji gte*). Uji calon baru dengan cara yang sama: 20 kandidat RRF asli untuk beasiswa, larangan kode etik, dan satu pertanyaan tak terjawab. `RERANK_CANDIDATES` yang lebih kecil bukan jalan keluar untuk T9, karena h27 ada di RRF#11. Setelah itu isi `RERANK_API_KEY` dan `RERANK_ENABLED=true` di `api/.env` (`api/docs/rerank.md` bagian *Menyalakan*). Mulai dengan `RERANK_THRESHOLD` kosong, lalu restart API.
2. Uji ulang di browser (cara menjalankan dan `ask.sh` ada di `browseract.md`), terutama pertanyaan yang ❌ di sana:
   - "Apa saja jenis beasiswa yang tersedia?" (Kemahasiswaan): pastikan h27/h4 naik ke 5 besar;
   - "Apa saja larangan bagi mahasiswa menurut kode etik?" (BAAK): Pasal 10 harus tetap di 3 besar.

   Pakai sesi baru tanpa riwayat (cara di `browseract.md` §4) supaya rewrite tidak mengubah query.

   Uji coba tidak menampilkan skor rerank. Pakai rincian giliran di halaman Log ("skor rerank"), atau skrip yang menilai 20 kandidat RRF secara langsung.
3. Jalankan eval retrieval (`python -m eval.run_eval`). Kalau reranker menyala, baris hybrid + RRF + rerank ikut dihitung. Bandingkan hasilnya dengan hasil tanpa reranker.
4. Ukur tambahan latensinya: satu panggilan per pertanyaan, maksimal `RERANK_TIMEOUT_SECONDS=10`. Catat di sini (bukan di `browseract.md`).
5. Kalibrasi `RERANK_THRESHOLD` dari `messages.meta.top_rerank_score` (lihat `api/docs/rerank.md` bagian *Ambang*). Setelah terisi, uji ulang T3.
6. Temuan yang masih terbuka dipindah kembali ke `browseract.md` §5, lalu perbarui status di file ini.
