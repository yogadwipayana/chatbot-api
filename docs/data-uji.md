# Panduan menghapus data uji

Skrip: `scripts/hapus_data_uji.py`. Terakhir diperbarui 2026-09-29.

## Kenapa perlu

Database pengembangan dan database produksi saat ini **satu database yang sama**: kontainer `pg` (`appdb`) di host server. Tailscale (`100.111.178.48:5432`), tunnel SSH, dan API di server (`localhost:5432`) semuanya menjangkau database itu.

Akibatnya, setiap percakapan uji ikut masuk ke data yang nanti dilihat admin:
- **Statistik** menghitungnya sebagai pertanyaan mahasiswa.
- **Pertanyaan tak terjawab** (AD-4) menampilkan penolakan uji ("resep rendang", "asdf qwer", upaya manipulasi) seolah itu celah dokumen yang perlu ditambal.

Jadi data uji harus dihapus sebelum chatbot dipakai mahasiswa, dan sebaiknya juga setiap selesai sesi uji besar.

## Apa yang dihapus dan apa yang tidak

Skrip memilih data per **sesi percakapan** (`conversations.session_id`). Untuk setiap sesi terpilih, dalam satu transaksi:
1. Entri **Pertanyaan tak terjawab** yang berasal dari sesi itu dihapus lebih dulu. Relasinya `ON DELETE SET NULL`, jadi kalau percakapan dihapus lebih dulu, entri ini tertinggal tanpa pesan asalnya dan tetap tampil di dashboard.
2. **Percakapan** sesi itu dihapus. **Pesan** dan **umpan balik**nya ikut terhapus otomatis (`CASCADE`).

Yang **tidak pernah** disentuh:
- Dokumen, potongan dokumen, entri tanya jawab, unit, akun admin, kunci sematan, dan konfigurasi.
- **Log aplikasi** (`log/app.db`, halaman Log admin). Log ini SQLite di mesin masing-masing, bukan di database bersama, dan otomatis dihapus setelah 7 hari (`LOG_RETENTION_DAYS`).
- **Trace LangSmith.** Tersimpan di LangSmith, bukan di database.
- **Data `seed_demo`.** Punya perintahnya sendiri: `python -m scripts.seed_demo --hapus`.
- **Uji coba admin** (halaman Uji coba) memang tidak pernah dicatat, jadi tidak ada yang perlu dihapus.

## Langkah

Semua perintah dijalankan dari folder `api/`.

### 1. Lihat semua sesi

```bash
.venv/Scripts/python -m scripts.hapus_data_uji
```

Tanpa target, skrip hanya menampilkan daftar dan **tidak menghapus apa pun**. Untuk setiap sesi ditampilkan jumlah percakapan, pertanyaan, jawaban, entri tak terjawab, umpan balik, rentang waktu, dan dua contoh pertanyaan. Baris pertama menyebut database yang dituju (nama dan host) serta zona waktu jam yang ditampilkan (`TIMEZONE` aplikasi, bawaan Asia/Jakarta, jadi satu jam di belakang WITA). Pastikan database-nya benar sebelum lanjut.

### 2. Pratinjau sesi yang akan dihapus

Pilih target dengan salah satu atau gabungan:
- `--sesi <session_id>`: nama sesi persis. Boleh diulang.
- `--awalan <teks>`: semua sesi yang namanya diawali teks itu. Minimal 4 karakter. `_` dan `%` dibaca apa adanya, bukan wildcard.

```bash
.venv/Scripts/python -m scripts.hapus_data_uji --awalan jev-uji- --sesi 82360357-f1af-4c88-b49b-9f71395ad4c2
```

Tanpa `--hapus`, ini tetap **pratinjau**: daftar sesi yang cocok beserta totalnya. Periksa contoh pertanyaannya. Kalau ada sesi yang tidak dikenal, jangan dilanjutkan.

### 3. Hapus

Perintah yang sama ditambah `--hapus`:

```bash
.venv/Scripts/python -m scripts.hapus_data_uji --awalan jev-uji- --sesi 82360357-f1af-4c88-b49b-9f71395ad4c2 --hapus
```

Skrip mencetak jumlah yang terhapus, lalu memeriksa ulang. Kalau tertulis `Pemeriksaan ulang: 0 sesi target tersisa`, penghapusan berhasil. Kode keluar 0 berarti bersih, 1 berarti masih ada sisa.

Tambahkan `--yatim` untuk sekaligus membersihkan entri tak terjawab yang pesan asalnya sudah tidak ada. Jumlah entri seperti ini selalu ditampilkan di mode daftar (langkah 1).

### 4. Periksa di dashboard

Buka halaman **Pertanyaan tak terjawab** dan **Statistik** di admin (:3000). Entri dan giliran uji seharusnya sudah hilang.

## Bersih total sebelum rilis

Kalau seluruh percakapan di database adalah data uji (belum ada mahasiswa sungguhan):

```bash
.venv/Scripts/python -m scripts.hapus_data_uji --semua              # pratinjau semuanya
.venv/Scripts/python -m scripts.hapus_data_uji --semua --hapus --konfirmasi HAPUS-SEMUA
```

`--semua --hapus` ditolak tanpa `--konfirmasi HAPUS-SEMUA`, dan `--semua` tidak bisa digabung dengan `--sesi` atau `--awalan`. **Jangan jalankan ini setelah chatbot dipakai mahasiswa**: percakapan mereka ikut terhapus.

## Riwayat pembersihan

**2026-09-29:** tiga sesi di bawah ini dihapus (`--awalan jev-uji- --sesi 82360357-f1af-4c88-b49b-9f71395ad4c2 --hapus`). Pemeriksaan ulang: 0 sisa, dan halaman *Pertanyaan tak terjawab* kosong. Widget di profil browser-act tetap memakai sesi `82360357-…`, jadi uji berikutnya dari profil itu dapat dihapus dengan perintah yang sama.

| Sesi | Asal | Percakapan | Jawaban | Tak terjawab |
|---|---|---|---|---|
| `82360357-f1af-4c88-b49b-9f71395ad4c2` | Uji widget :3001 lewat browser-act (profil `github-personal`) | 3 | 47 | 16 |
| `jev-uji-on-1790599167` | Uji A/B JEV aktif (`jev.md` di root repo) | 1 | 23 | 4 |
| `jev-uji-off-1790599480` | Uji A/B JEV nonaktif | 1 | 23 | 14 |
| **Total** | | **5** | **93** | **34** |

Sesi `4399d35b-0c25-4511-bdca-0b8605548bb4` ("bagaimana cara membuat lpj tahunan UKM", 28/9) bukan dari sesi uji Claude. Sengaja tidak dimasukkan ke perintah di atas. Kalau itu juga uji, tambahkan `--sesi 4399d35b-0c25-4511-bdca-0b8605548bb4`.

## Menemukan sesi uji sendiri

Widget mahasiswa menyimpan `session_id` acak di `localStorage` peramban dengan kunci `chatbot:session-id` (`../client/src/lib/session.ts`). Satu profil peramban memakai satu sesi yang sama, termasuk untuk percakapan-percakapan berikutnya.

- **Cek di peramban:** buka DevTools di :3001, lalu jalankan `localStorage.getItem("chatbot:session-id")`.
- **Cek lewat browser-act:**
  ```bash
  browser-act --session local3001 eval "localStorage.getItem('chatbot:session-id')"
  ```
- **Tidak tahu sesinya?** Pakai mode daftar (langkah 1) dan cocokkan dengan waktu serta contoh pertanyaan.

**Supaya pembersihan mudah:** uji otomatis lewat API sebaiknya memakai `session_id` dengan awalan yang jelas, misalnya `jev-uji-…` seperti skrip A/B di `jev.md`. Hindari angka acak yang mirip sesi mahasiswa.

## Kalau gagal

- **Tidak bisa terhubung ke database:** cek bahwa Tailscale aktif (`ping 100.111.178.48`) dan `DATABASE_URL` di `.env` benar. Cadangannya adalah tunnel SSH di `start.md` atau `start.txt` di root repo, dengan `DATABASE_URL` diarahkan ke `localhost`.
- **`error: awalan terlalu pendek`:** pakai awalan yang lebih spesifik, atau `--sesi` dengan nama lengkap.
- **`Pemeriksaan ulang` masih menemukan sisa:** jalankan pratinjau lagi. Kemungkinan ada percakapan baru di sesi yang sama saat skrip berjalan (widget masih terbuka). Tutup widgetnya, lalu ulangi.
