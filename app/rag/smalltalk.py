"""Sapaan dan basa-basi (bukan pertanyaan administrasi).

"hai" tidak punya jawaban di dokumen resmi mana pun. Tanpa penanganan khusus,
pesan seperti itu menempuh seluruh alur retrieval lalu keluar sebagai penolakan
FR-3 -- mahasiswa yang baru menyapa langsung disuruh datang ke loket biro. Baris
itu juga masuk tabel `unanswered_questions` dan mengotori AD-4 dengan "pertanyaan" yang
tidak pernah berupa pertanyaan.

Pemeriksaan ini berjalan SETELAH FR-7: "halo, saya stres berat" harus tetap
ditangani sebagai pertanyaan sensitif, bukan dibalas sapaan basa-basi.

Deteksinya sengaja berupa daftar kata, bukan LLM: dapat dijelaskan, tidak
berbiaya, dan tidak menambah latency. Syaratnya ketat -- SELURUH pesan harus
berupa basa-basi dan pendek -- supaya pertanyaan sungguhan tidak pernah tertelan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class SmallTalkKind(StrEnum):
    NONE = "none"
    GREETING = "greeting"
    THANKS = "thanks"
    CLOSING = "closing"


REPLIES: dict[SmallTalkKind, str] = {
    SmallTalkKind.GREETING: (
        "Halo! Saya asisten administrasi akademik INSTIKI. Silakan tanyakan urusan "
        "administrasi Anda -- jawaban saya bersumber dari dokumen resmi kampus "
        "dan selalu menyertakan sumbernya."
    ),
    SmallTalkKind.THANKS: (
        "Sama-sama. Kalau masih ada yang ingin ditanyakan soal administrasi "
        "akademik, silakan."
    ),
    SmallTalkKind.CLOSING: (
        "Baik, semoga urusannya lancar. Silakan kembali bertanya kapan saja."
    ),
}

MAKS_KATA_INTI = 3
"""Di atas ini pesan dianggap punya maksud lain, sependek apa pun basa-basinya."""

_KATA = re.compile(r"[a-z]+")

TAWA = re.compile(
    r"^(?:(?:wk){2,}w?|(?:ha){2,}h?|(?:he){2,}h?|(?:hi){2,}h?|(?:hu){2,}h?|(?:xi){2,}|lol|(?:a?wok){2,})$"
)
"""Tawa ("wkwk", "hehe", "awokawok"). Diabaikan di sini, seperti sapaan
pelengkap: "wkwk oke makasih" adalah ucapan terima kasih. Pesan yang HANYA
berisi tawa ditangani `app.rag.rule_gate`."""

# Sapaan dan panggilan yang hanya menempel pada maksud sebenarnya: "halo min",
# "makasih ya kak". Dibuang lebih dulu supaya tidak menghabiskan jatah kata inti.
_PELENGKAP = frozenset(
    {
        "min",
        "admin",
        "kak",
        "kakak",
        "bu",
        "ibu",
        "pak",
        "bapak",
        "bang",
        "gan",
        "bot",
        "ya",
        "yaa",
        "dong",
        "deh",
        "nih",
        "banget",
        "selamat",
        "mohon",
        "maaf",
    }
)

# Kata yang lazim mengiringi ucapan terima kasih atau penutup, tetapi tidak
# pernah membawa permintaan sendiri: "oke siap min, nanti saya coba dulu ya",
# "baik, sudah jelas penjelasannya", "sip, itu saja dulu pertanyaan saya".
# Tidak dihitung sebagai kata inti -- tanpa ini ketiga pesan itu melewati
# MAKS_KATA_INTI, lolos ke LLM, lalu masuk AD-4 sebagai "tidak ditemukan".
# Pesan yang HANYA berisi kata netral ("itu saja") tetap tidak dianggap
# basa-basi: harus ada setidaknya satu kata kunci.
_NETRAL = frozenset(
    {
        "saya",
        "aku",
        "kami",
        "itu",
        "ini",
        "dulu",
        "nanti",
        "sudah",
        "udah",
        "saja",
        "aja",
        "jelas",
        "paham",
        "mengerti",
        "ngerti",
        "coba",
        "dicoba",
        "kalau",
        "gitu",
        "begitu",
        "atas",
        "untuk",
        "sangat",
        "membantu",
        "infonya",
        "informasinya",
        "penjelasannya",
        "jawabannya",
        "bantuannya",
        "pertanyaan",
        "pertanyaannya",
        "semoga",
        "sehat",
        "selalu",
        "sukses",
        "lancar",
        "juga",
        "hari",
        "harimu",
        "menyenangkan",
    }
)

_PRIORITAS = (SmallTalkKind.THANKS, SmallTalkKind.CLOSING, SmallTalkKind.GREETING)
"""Pesan campuran dibalas menurut jenis terkuat: "oke, makasih" dijawab
"sama-sama", dan "pagi min, oke siap" dijawab sebagai penutup, bukan sapaan
pembuka -- mahasiswa yang berterima kasih atau pamit tidak sedang memulai."""

_FRASA: dict[str, SmallTalkKind] = {
    "terima kasih": SmallTalkKind.THANKS,
    "terima kasih banyak": SmallTalkKind.THANKS,
    "makasih banyak": SmallTalkKind.THANKS,
    "matur suksma": SmallTalkKind.THANKS,
    # Salam pembuka Bali, lazim di pengumuman dan acara kampus INSTIKI.
    "om swastyastu": SmallTalkKind.GREETING,
    "om swastiastu": SmallTalkKind.GREETING,
    "thank you": SmallTalkKind.THANKS,
    "sampai jumpa": SmallTalkKind.CLOSING,
    "sudah cukup": SmallTalkKind.CLOSING,
}

_KATA_KUNCI: dict[str, SmallTalkKind] = {
    # Sapaan
    "hai": SmallTalkKind.GREETING,
    "hi": SmallTalkKind.GREETING,
    "halo": SmallTalkKind.GREETING,
    "hallo": SmallTalkKind.GREETING,
    "helo": SmallTalkKind.GREETING,
    "hello": SmallTalkKind.GREETING,
    "hey": SmallTalkKind.GREETING,
    "pagi": SmallTalkKind.GREETING,
    "siang": SmallTalkKind.GREETING,
    "sore": SmallTalkKind.GREETING,
    "malam": SmallTalkKind.GREETING,
    "assalamualaikum": SmallTalkKind.GREETING,
    "assalamualaikum warahmatullahi": SmallTalkKind.GREETING,
    "salam": SmallTalkKind.GREETING,
    "permisi": SmallTalkKind.GREETING,
    "punten": SmallTalkKind.GREETING,
    "misi": SmallTalkKind.GREETING,
    "swastyastu": SmallTalkKind.GREETING,
    "swastiastu": SmallTalkKind.GREETING,
    # Terima kasih
    "makasih": SmallTalkKind.THANKS,
    "makasi": SmallTalkKind.THANKS,
    "terimakasih": SmallTalkKind.THANKS,
    "thanks": SmallTalkKind.THANKS,
    "thx": SmallTalkKind.THANKS,
    "trims": SmallTalkKind.THANKS,
    "tengkyu": SmallTalkKind.THANKS,
    "suksma": SmallTalkKind.THANKS,
    "nuhun": SmallTalkKind.THANKS,
    # Penutup
    "oke": SmallTalkKind.CLOSING,
    "ok": SmallTalkKind.CLOSING,
    "okey": SmallTalkKind.CLOSING,
    "sip": SmallTalkKind.CLOSING,
    "siap": SmallTalkKind.CLOSING,
    "baik": SmallTalkKind.CLOSING,
    "mantap": SmallTalkKind.CLOSING,
    "bye": SmallTalkKind.CLOSING,
    "dadah": SmallTalkKind.CLOSING,
    "okay": SmallTalkKind.CLOSING,
    "okee": SmallTalkKind.CLOSING,
    "noted": SmallTalkKind.CLOSING,
    "cukup": SmallTalkKind.CLOSING,
    "sekian": SmallTalkKind.CLOSING,
}

# Penjaga terakhir: apa pun yang berbau pertanyaan tidak pernah dianggap
# basa-basi, walau sependek "oke kapan?".
_TANDA_PERTANYAAN = re.compile(
    r"\?|\b(?:apa|apakah|kapan|bagaimana|gimana|berapa|kenapa|mengapa|mana|"
    r"siapa|cara|syarat|bisakah|boleh|tolong)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SmallTalkAssessment:
    kind: SmallTalkKind
    reply: str = ""

    @property
    def handled(self) -> bool:
        """True berarti retrieval dan LLM dilewati; balasan singkat yang dikirim."""
        return self.kind is not SmallTalkKind.NONE


_TIDAK_ADA = SmallTalkAssessment(SmallTalkKind.NONE)


def detect(text: str) -> SmallTalkAssessment:
    """Kenali pesan yang seluruhnya berupa sapaan, ucapan terima kasih, atau penutup."""
    if _TANDA_PERTANYAAN.search(text):
        return _TIDAK_ADA

    jenis = _jenis_per_unit(_kata_inti(text))
    # Satu kata tak dikenal (None) sudah cukup untuk membatalkan: "halo
    # skripsi" adalah awal pertanyaan, bukan sapaan.
    if not jenis or len(jenis) > MAKS_KATA_INTI or None in jenis:
        return _TIDAK_ADA
    kind = _utama(jenis)
    return SmallTalkAssessment(kind, REPLIES[kind])


def reply_for(text: str) -> str | None:
    """Balasan basa-basi yang cocok untuk pesan yang SUDAH divonis smalltalk
    (oleh gerbang JEV), atau None bila tidak ada kata yang dikenali.

    Longgar, berbeda dari `detect`: kata tak dikenal diabaikan, karena yang
    diputuskan di sini hanya nada balasannya, bukan apakah pesan itu pertanyaan.
    Tanpa ini JEV membalas "sip, itu saja dulu" dengan sapaan pembuka.
    """
    dikenal = [k for k in _jenis_per_unit(_kata_inti(text)) if k is not None]
    return REPLIES[_utama(dikenal)] if dikenal else None


def _kata_inti(text: str) -> list[str]:
    return [
        kata
        for kata in _KATA.findall(text.lower())
        if kata not in _PELENGKAP and kata not in _NETRAL and not TAWA.match(kata)
    ]


def _jenis_per_unit(kata: list[str]) -> list[SmallTalkKind | None]:
    """Satu jenis per frasa atau kata. Frasa dikenali di posisi mana pun, supaya
    "oke terima kasih" tidak gagal hanya karena "terima kasih" didahului kata lain."""
    hasil: list[SmallTalkKind | None] = []
    i = 0
    while i < len(kata):
        for n in (3, 2):
            frasa = " ".join(kata[i : i + n])
            if len(kata) - i >= n and frasa in _FRASA:
                hasil.append(_FRASA[frasa])
                i += n
                break
        else:
            hasil.append(_KATA_KUNCI.get(kata[i]))
            i += 1
    return hasil


def _utama(jenis: list[SmallTalkKind | None]) -> SmallTalkKind:
    return next(k for k in _PRIORITAS if k in jenis)
