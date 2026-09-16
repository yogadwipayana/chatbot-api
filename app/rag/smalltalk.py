"""Sapaan dan basa-basi (bukan pertanyaan administrasi).

"hai" tidak punya jawaban di dokumen resmi mana pun. Tanpa penanganan khusus,
pesan seperti itu menempuh seluruh alur retrieval lalu keluar sebagai penolakan
FR-3 -- mahasiswa yang baru menyapa langsung disuruh datang ke loket biro. Baris
itu juga masuk tabel `unanswered` dan mengotori AD-4 dengan "pertanyaan" yang
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
        "Halo! Saya asisten administrasi akademik. Silakan tanyakan urusan "
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

_FRASA: dict[str, SmallTalkKind] = {
    "terima kasih": SmallTalkKind.THANKS,
    "terima kasih banyak": SmallTalkKind.THANKS,
    "makasih banyak": SmallTalkKind.THANKS,
    "matur suksma": SmallTalkKind.THANKS,
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

    inti = [kata for kata in _KATA.findall(text.lower()) if kata not in _PELENGKAP]
    if not inti or len(inti) > MAKS_KATA_INTI:
        return _TIDAK_ADA

    kind = _FRASA.get(" ".join(inti))
    if kind is None:
        jenis = {_KATA_KUNCI.get(kata) for kata in inti}
        # Satu kata tak dikenal (None ikut masuk himpunan) sudah cukup untuk
        # membatalkan: "halo skripsi" adalah awal pertanyaan, bukan sapaan.
        kind = jenis.pop() if len(jenis) == 1 else None

    if kind is None:
        return _TIDAK_ADA
    return SmallTalkAssessment(kind, REPLIES[kind])
