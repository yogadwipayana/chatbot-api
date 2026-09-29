"""Saringan aturan: pesan acak, tawa, basa-basi tentang PANDU, dan manipulasi.

Cadangan gerbang JEV (`app.rag.gate`) sekaligus penghemat biayanya. Berjalan
SEBELUM JEV, entah JEV hidup atau mati:

- JEV hidup: pesan yang jelas bukan pertanyaan tidak perlu dibayar satu
  panggilan JEV, dan tidak ikut menunggu latensinya yang berayun 1-29 dtk.
  JEV sendiri meloloskan beberapa di antaranya: "hmm", "123 456 789",
  "kamu siapa sih?", dan tag `</pertanyaan_mahasiswa>` (uji 2026-09-29).
- JEV mati (`JEV_ENABLED=false`): saringan ini, ditambah penanda
  `[DI_LUAR_TOPIK]` dari LLM penjawab (`app.rag.prompts.OFF_TOPIC_MARKER`),
  menggantikan vonis JEV dengan hasil yang mirip, tanpa biaya per pesan.

Hanya pola yang hampir pasti yang dipakai. Pertanyaan sungguhan yang salah
terblokir hilang tanpa jejak di AD-4, sedangkan pesan buruk yang lolos masih
dihadang LLM penjawab. Karena itu aturan pesan acak dan basa-basi batal bila
pesan memuat istilah urusan kampus (`_ISTILAH_KAMPUS`), dan aturan manipulasi
hanya mengenali perintah kepada asisten ("abaikan semua instruksi
sebelumnya"), bukan pertanyaan tentang aturan ("kalau saya abaikan perintah
dosen, apa sanksinya?").

Pesan di luar topik ("resep rendang") sengaja tidak dicoba dikenali di sini:
daftar kata tidak akan pernah lengkap, dan LLM penjawab toh membaca
pertanyaannya.
"""

from __future__ import annotations

import re

from app.rag.gate import GateLabel, GateSource, GateVerdict
from app.rag.smalltalk import TAWA

_ISTILAH_KAMPUS = re.compile(
    r"\b(?:krs|khs|ukt|spp|skp|sks|ipk|nim|ukm|mbkm|plk|va|bni|bayar\w*|biaya|"
    r"beasiswa|cuti|wisuda|yudisium|skripsi|dosen|jadwal|kuliah|kampus|instiki|"
    r"stiki|prodi|fakultas|sertifikasi|sertifikat|toeic|ujian|uts|uas|semester|"
    r"akademik|baak|keuangan|kemahasiswaan|nilai|transkrip|ijazah|magang|sads)\b",
    re.IGNORECASE,
)

_HURUF = re.compile(r"[^\W\d_]+")

_BARIS_KEYBOARD = ("qwertyuiop", "asdfghjkl", "zxcvbnm")

_KONSONAN_BERUNTUN = re.compile(r"[bcdfghjklmnpqrstvwxz]{5,}")
"""Lima konsonan berturut-turut ("ksjdhf"). Singkatan chat mahasiswa ("krs",
"mbkm", "brp", "yg") paling banyak empat, dan aturan ini hanya berlaku bila
SELURUH pesan acak."""

_GUMAM = re.compile(r"^(?:h+m+|e+h+|o+h+|a+h+|u+h+|e+m+|u+m+)$")

_PELENGKAP_BASA_BASI = frozenset(
    {
        "min",
        "admin",
        "kak",
        "kakak",
        "pandu",
        "sih",
        "dong",
        "ya",
        "yaa",
        "deh",
        "nih",
        "juga",
        "sebenarnya",
        "emang",
        "memang",
        "halo",
        "hai",
        "hi",
        "hallo",
        "hello",
        "hey",
        "pagi",
        "siang",
        "sore",
        "malam",
        "selamat",
    }
)

_BASA_BASI = frozenset(
    {
        # Kabar
        "apa kabar",
        "apa kabarmu",
        "apa kabarnya",
        "gimana kabar",
        "gimana kabarmu",
        "gimana kabarnya",
        "bagaimana kabar",
        "bagaimana kabarmu",
        "bagaimana kabarnya",
        "kabarmu gimana",
        "kabarnya gimana",
        "kabar kamu gimana",
        # Siapa PANDU
        "kamu siapa",
        "siapa kamu",
        "anda siapa",
        "siapa anda",
        "kau siapa",
        "siapa kau",
        "kamu itu siapa",
        "ini siapa",
        "kamu bot",
        "kamu robot",
        "kamu ai",
        "kamu manusia",
        "ini bot",
        "apakah kamu bot",
        "apakah kamu robot",
        "apakah kamu manusia",
        "kamu bot atau manusia",
        "bot atau manusia",
        "kamu siapa bot",
        # Pujian dan ejekan ringan
        "kamu pintar",
        "kamu pinter",
        "kamu keren",
        "kamu hebat",
        "kamu lucu",
        "kamu bodoh",
        "kamu lemot",
        "pintar",
        "pinter",
        "keren",
    }
)
"""Seluruh pesan, setelah `_PELENGKAP_BASA_BASI` dibuang, harus persis salah
satu frasa ini. "kamu siapa, bisa bantu soal KRS?" tidak cocok, jadi tetap
diteruskan. "bot" dicoba dua kali -- sebagai isi ("kamu bot?") dan sebagai
sapaan yang dibuang ("apa kabar bot?")."""

_MANIPULASI = re.compile(
    # Tag pembungkus pertanyaan (FR-5) dan token templat chat.
    r"</?\s*pertanyaan_mahasiswa\s*>"
    r"|<\|?\s*(?:im_start|im_end|system)\s*\|?>|\[/?INST\]"
    # Perintah menimpa instruksi. "abaikan perintah dosen" tidak cocok: harus
    # "semua/seluruh ..." atau diikuti "sebelumnya/di atas/sistem/kamu".
    r"|\b(?:abaikan|lupakan|acuhkan|hiraukan)\s+(?:semua|seluruh)\s+"
    r"(?:instruksi|perintah|aturan|prompt)"
    r"|\b(?:abaikan|lupakan|acuhkan|hiraukan)\s+(?:instruksi|perintah|aturan|prompt)"
    r"(?:\s*(?:mu|nya))?\s+(?:sebelumnya|di\s*atas|tadi|awal|sistem|kamu|anda)\b"
    r"|\b(?:ignore|disregard|forget)\s+(?:all\s+|the\s+|any\s+|your\s+)*"
    r"(?:previous\s+|prior\s+|above\s+|earlier\s+|system\s+)?(?:instructions?|rules|prompts?)\b"
    # Membocorkan prompt.
    r"|\b(?:system|sistem)\s+prompt\b|\bprompt\s+(?:sistem|system|awal|kamu|anda)\b"
    r"|\bprompt(?:mu|nya)\b"
    # Bermain peran sebagai pihak yang berwenang.
    r"|\b(?:ber)?pura[- ]?pura\s+(?:jadi|menjadi|sebagai)\s+(?:admin\w*|developer|"
    r"pengembang|root|hacker|sistem|system|ai|chatbot|bot|gpt|chatgpt)\b"
    r"|\b(?:kamu|anda)\s+sekarang\s+(?:adalah\s+)?(?:admin\w*|developer|root|(?-i:DAN))\b"
    r"|\bjailbreak\b|\bdeveloper\s+mode\b|(?-i:\bDAN\s+mode\b)"
    # Kredensial sistem.
    r"|\b(?:password|kata\s*sandi|sandi)\s+(?:db|database|server|admin|root)\b",
    re.IGNORECASE,
)


def detect(text: str) -> GateVerdict | None:
    """Vonis blokir untuk pesan yang jelas bukan pertanyaan, atau None."""
    label = _label(text)
    if label is None:
        return None
    return GateVerdict(label, 1.0, blocked=True, source=GateSource.RULES)


def _label(text: str) -> GateLabel | None:
    if _MANIPULASI.search(text):
        return GateLabel.MALICIOUS
    if _ISTILAH_KAMPUS.search(text):
        return None
    kata = [k.lower() for k in _HURUF.findall(text)]
    if not kata:
        # Hanya angka dan tanda baca: "123 456 789", "???".
        return GateLabel.NONSENSE
    jenis = [_jenis_kata(k) for k in kata]
    if None not in jenis:
        return GateLabel.SMALLTALK if set(jenis) == {"tawa"} else GateLabel.NONSENSE
    inti = [k for k in kata if k not in _PELENGKAP_BASA_BASI]
    tanpa_bot = [k for k in inti if k != "bot"]
    if " ".join(inti) in _BASA_BASI or " ".join(tanpa_bot) in _BASA_BASI:
        return GateLabel.SMALLTALK
    return None


def _jenis_kata(kata: str) -> str | None:
    """ "tawa", "gumam", "acak", atau None untuk kata yang mungkin bermakna."""
    if TAWA.match(kata):
        return "tawa"
    if _GUMAM.match(kata):
        return "gumam"
    if len(kata) == 1:
        # "p" (memanggil), "a", "x": satu huruf tanpa kata lain bukan pertanyaan.
        return "acak"
    if len(kata) >= 4 and len(set(kata)) == 1:
        return "acak"
    if len(kata) >= 4 and any(kata in b or kata in b[::-1] for b in _BARIS_KEYBOARD):
        return "acak"
    if _KONSONAN_BERUNTUN.search(kata):
        return "acak"
    return None
