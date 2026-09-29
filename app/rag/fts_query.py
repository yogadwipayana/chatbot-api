"""Query fulltext yang dibentuk dari pertanyaan mahasiswa (FR-2).

`websearch_to_tsquery` menggabungkan setiap kata dengan DAN, dan konfigurasi
Postgres `indonesian` tidak punya daftar stopword. Pertanyaan utuh seperti
"berapa harga sertifikasi TOEIC?" menjadi `'apa' & 'harga' & 'sertifikasi' &
'toeic'`, dan tidak ada satu pun potongan yang memuat keempatnya. Diukur
2026-09-29: enam pertanyaan uji semuanya 0 potongan, padahal "toeic" saja
menemukan 2. Jalur fulltext praktis mati, dan pencarian "hibrida" hanya vektor.

Di sini kata tanya, kata sambung, dan sapaan dibuang, lalu sisanya digabung
dengan `or`: "harga or sertifikasi or toeic". Peringkat tetap `ts_rank` di SQL
(`app.rag.retriever.fulltext_sql`) -- potongan yang memuat lebih banyak kata
pertanyaan naik ke atas.

Penyaringan memakai kata MENTAH, sebelum stemmer Postgres. Stemmer
menggabungkan stopword dengan kata isi (permisi -> misi, peserta -> serta,
permohonan -> mohon, melalui -> lalu), jadi daftar di tingkat stem diam-diam
akan menghapus istilah penting.

Teks keluaran tetap diproses `websearch_to_tsquery`, yang tidak pernah melempar
galat sintaks. Token diambil dengan pola kata, jadi tanda kutip, tanda minus
(NOT), dan "or" yang diketik mahasiswa tidak ikut menjadi operator.
"""

from __future__ import annotations

import re

from app.rag.glossary import cari_istilah, fulltext_variants

STOPWORDS: frozenset[str] = frozenset(
    {
        # Kata tanya, termasuk singkatan chat. Bentuk -kah/-pun ditangani _KLITIK.
        "apa", "apaan", "siapa", "kapan", "mana", "dimana", "kemana", "darimana",
        "bagaimana", "gimana", "gmn", "bgmn", "berapa", "brp", "mengapa", "kenapa",
        "knp", "napa", "bilamana",
        # Kata sambung dan kata depan. "dan" ada di 75% potongan, "yang" 62%.
        "yang", "yg", "dan", "di", "ke", "dari", "untuk", "utk", "dengan", "dgn",
        "atau", "ini", "itu", "tsb", "ada", "adalah", "ialah", "merupakan", "pada",
        "dalam", "akan", "sudah", "udah", "sdh", "telah", "belum", "blm", "masih",
        "juga", "jg", "saja", "aja", "sja", "pun", "kah", "lah", "nya", "sebagai",
        "oleh", "tentang", "terhadap", "bagi", "antara", "secara", "karena", "karna",
        "krn", "jika", "kalau", "kalo", "klo", "apabila", "bila", "maka", "agar",
        "supaya", "sehingga", "tetapi", "tapi", "namun", "serta", "hingga", "sampai",
        "sejak", "setelah", "sesudah", "sebelum", "selama", "saat", "ketika", "para",
        "sang", "si", "tidak", "tak", "gak", "ga", "nggak", "ngga", "enggak", "engga",
        "bukan", "sangat", "lebih", "paling", "semua", "setiap", "tiap", "hal",
        "tersebut", "hanya", "meski", "walau", "maupun", "bahwa", "seperti",
        "mengenai", "terkait", "seputar", "menurut", "soal", "lalu", "kemudian",
        "terus", "trus", "gitu", "begitu", "melalui", "lewat", "via", "buat", "lagi",
        "dll", "dsb", "dst",
        # Operator websearch dalam bahasa Inggris.
        "or", "and",
        # Modalitas dan kata permintaan.
        "dapat", "harus", "perlu", "bisa", "boleh", "mau", "ingin", "pengen",
        "pingin", "hendak", "mohon", "tolong", "minta", "punya", "info", "tanya",
        "nanya", "bertanya", "jelaskan", "sebutkan",
        # Kata ganti.
        "saya", "aku", "gue", "gw", "sy", "kami", "kita", "anda", "kamu", "dia",
        "ia", "mereka", "beliau", "ku", "mu",
        # Sapaan dan partikel. "pagi", "siang", "sore" sengaja TIDAK di sini:
        # "kelas sore" dan jadwal kuliah pagi adalah pertanyaan sungguhan.
        "min", "mimin", "admin", "kak", "kakak", "bang", "pak", "bu", "mbak", "dong",
        "deh", "sih", "kok", "ya", "yah", "iya", "nih", "tuh", "kan", "loh", "lho",
        "halo", "hai", "hi", "hallo", "permisi", "selamat", "terima", "kasih",
        "makasih", "thanks", "thx",
        # Pembingkai pertanyaan: "bagaimana cara X" menanyakan X, dan potongan
        # jawabannya jarang memuat kata "cara".
        "cara",
        # Penanda entri tanya jawab ("Pertanyaan: ... / Jawaban: ...") -- ada di
        # setiap potongan tanya jawab, jadi tidak membedakan apa pun.
        "pertanyaan", "jawaban", "jawab", "menjawab",
    }
)  # fmt: skip

_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*")
"""Kata; istilah bertanda hubung (KIP-K) tetap utuh."""

_KLITIK = re.compile(r"(?:nya|kah|lah|pun)$")
"""Hanya untuk MENCARI di STOPWORDS: apakah, caranya, infonya, adalah. Kata yang
dikirim ke query tidak pernah diubah -- stemmer Postgres yang mengurusnya."""


def _kata_bermakna(teks: str) -> list[str]:
    hasil = []
    for token in _TOKEN.findall(teks.casefold()):
        if token in STOPWORDS or _KLITIK.sub("", token) in STOPWORDS:
            continue
        # Huruf tunggal dan angka satu digit: "1", "2", "3" ada di 40-47% potongan.
        if len(token) == 1:
            continue
        hasil.append(token)
    return hasil


def fulltext_query(teks: str) -> str:
    """"berapa harga sertifikasi TOEIC?" -> "harga or sertifikasi or toeic".

    Istilah kamus kampus dikirim utuh -- yang multi-kata sebagai frasa berkutip,
    supaya "Pembelajaran di Luar Kampus" tidak terpecah menjadi kata umum dan
    "di" di dalamnya tidak ikut dibuang. String kosong bila semua kata stopword.
    """
    bagian: list[str] = []
    awal = 0
    for cocok in cari_istilah(teks):
        bagian += _kata_bermakna(teks[awal : cocok.start()])
        istilah = " ".join(_TOKEN.findall(cocok.group(0).casefold()))
        bagian.append(f'"{istilah}"' if " " in istilah else istilah)
        awal = cocok.end()
    bagian += _kata_bermakna(teks[awal:])
    return " or ".join(dict.fromkeys(bagian))


def fulltext_queries(pertanyaan: str) -> list[str]:
    """Satu query `or` per varian kamus sinonim, tanpa duplikat.

    Kamus dijalankan LEBIH DULU pada teks mentah (polanya mencocokkan istilah
    yang memuat stopword, mis. "Institut Bisnis dan Teknologi Indonesia"), baru
    stopword dibuang per varian. Daftar kosong berarti tidak ada kata yang layak
    dicari: pencarian fulltext dilewati dan vektor yang memutuskan.
    """
    hasil: list[str] = []
    for varian in fulltext_variants(pertanyaan):
        query = fulltext_query(varian)
        if query and query not in hasil:
            hasil.append(query)
    return hasil
