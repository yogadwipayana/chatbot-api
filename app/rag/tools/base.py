"""Primitif tool-calling: spesifikasi, hasil, validasi argumen, kartu sitasi.

Murni data + fungsi; tidak mengimpor LangGraph maupun pipeline, supaya mudah
diuji dan dipakai ulang antar layanan. Lihat `docs/tool-call.md` (§6, §10, §11).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.db.models import DocumentType

MAKS_PANJANG_ARGUMEN = 200
"""Batas panjang nilai argumen string dari model sebelum diteruskan ke layanan.

Argumen yang dihalusinasi model paling buruk menghasilkan hasil kosong, bukan
beban atau panggilan tak terduga ke layanan hulu (docs/tool-call.md §11)."""


class ToolArgumentError(ValueError):
    """Argumen dari model tidak memenuhi skema tool."""


@dataclass(frozen=True)
class Lampiran:
    """Daftar dari hasil tool yang ditampilkan widget apa adanya (docs/tool-call.md §10a).

    Model tidak menyalin daftar ini ke jawabannya. Daftar 220 dosen yang ditulis
    ulang model memakan sekitar 3,8 ribu token keluaran dan 30–70 detik, dan
    sekali disalin, setiap hitungan atau saringan atasnya dikerjakan model
    yang tidak andal menghitung (T46: "38 dosen bergelar Dr." dari 25)."""

    title: str
    """Judul kotak daftar, mis. "Dosen bergelar Dr."."""
    source: str
    """Penanda sitasi asal datanya. Lampiran hanya tampil bila penanda ini dikutip."""
    items: tuple[str, ...]
    disebut: str = ""
    """Nama yang harus ditulis jawaban supaya lampiran ini tampil, mis. "Basis Data".

    Kosong = cukup sumbernya dikutip. Dipakai bila satu panggilan tool
    menghasilkan beberapa lampiran yang tidak semuanya ditanyakan
    (`pilih_lampiran`), mis. "Basis Data" dan "Basis Data Lanjut"."""


@dataclass(frozen=True)
class ToolResult:
    """Hasil satu panggilan tool: teks untuk model + label kartu sitasi."""

    name: str
    label: str
    """Penanda sitasi, mis. "Data akademik SADS". Kosong bila gagal."""
    text: str
    ok: bool = True
    attachments: tuple[Lampiran, ...] = ()
    """Daftar yang ditampilkan langsung ke mahasiswa; kosong = model menulis
    jawabannya sendiri dari `text`, seperti sebelum lampiran ada."""

    def pesan_untuk_model(self) -> str:
        """Isi pesan `role:"tool"` yang dibalikkan ke model.

        Diberi awalan penanda sitasi supaya model tahu persis cara mengutipnya
        (aturan T3). Kegagalan dibalas penanda datar: model menjawab apa adanya
        atau mengakui data tidak tersedia, bukan mengarang (docs/tool-call.md §12).

        Bila ada lampiran, model diberi tahu bahwa daftarnya sudah tampil di bawah
        jawaban (aturan T5). Datanya tetap dikirim utuh: model masih butuh nama-
        namanya untuk menjawab, mis. "apakah Pak Totok dosen INSTIKI?"."""
        if not self.ok:
            return "DATA_TIDAK_TERSEDIA: data tidak dapat diambil saat ini."
        kepala = f"(Kutip data berikut dengan menyalin penanda [{self.label}].)"
        if self.attachments:
            kepala += f"\n{CATATAN_LAMPIRAN}"
        if sum(1 for a in self.attachments if a.disebut) > 1:
            kepala += f"\n{CATATAN_SEBUT}"
        return f"{kepala}\n\n{self.text}"


CATATAN_LAMPIRAN = (
    "(DAFTAR_DITAMPILKAN: daftar di bawah ditampilkan otomatis kepada mahasiswa "
    "tepat di bawah jawaban Anda. Jangan menyalin seluruh daftarnya, juga bila "
    "pertanyaannya meminta daftar; salin jumlahnya persis seperti tertulis di "
    "data, lalu jawab pertanyaannya secara ringkas. Butir tertentu boleh disebut "
    "bila pertanyaannya menanyakan butir itu, mis. 'apakah Pak Totok dosen "
    "INSTIKI?' atau 'apakah Pak Asroni mengajar Database?'.)"
)
"""Sisipan pesan `role:"tool"` untuk hasil berlampiran; dirujuk aturan T5.

Hanya hasil berlampiran yang membawanya. Aturan T5 yang berlaku umum ("jangan
menyalin, cukup jumlahnya") pernah membuat "siapa dosen pengampu Web
Programming?" -- tool tanpa lampiran -- dijawab "berjumlah 23 orang" tanpa satu
nama pun (uji live 2026-10-07).

Kalimat terakhirnya dulu "Nama tertentu boleh disebut bila pertanyaannya tentang
orang itu". Untuk `get_mk_dosen` setiap pertanyaan memang tentang satu orang,
sehingga "mata kuliah apa saja yang diajar Ahmad Asroni?" disalin model
lengkap 25 butir di atas lampiran yang sama (uji live 2026-10-07)."""

CATATAN_SEBUT = (
    "(DAFTAR_DITAMPILKAN: daftar setiap mata kuliah hanya ditampilkan bila nama "
    "mata kuliah itu Anda tulis di jawaban seperti tertulis di data. Tulis nama "
    "mata kuliah yang ditanyakan, dan jangan menulis nama mata kuliah lain yang "
    "tidak ditanyakan.)"
)
"""Sisipan tambahan untuk hasil dengan beberapa lampiran ber-`disebut` (T49).

`get_mk_diampu_dosen("Basis Data")` mengembalikan "Basis Data" dan "Basis Data
Lanjut", masing-masing berlampiran. Model memilih mata kuliah yang ditanyakan
dengan menuliskan namanya, dan `pilih_lampiran` hanya meneruskan lampiran yang
namanya tertulis."""


def kunci_sebutan(teks: str) -> str:
    """Samakan penulisan untuk mencocokkan nama: huruf kecil, spasi tunggal, huruf
    ganda dirapatkan ("Artificial Intelligence" = "artificial inteligence")."""
    return re.sub(r"(\w)\1", r"\1", " ".join(teks.split()).casefold())


def pilih_lampiran(lampiran: Sequence[Lampiran], jawaban: str) -> list[Lampiran]:
    """Lampiran dari SATU hasil tool yang ditampilkan di bawah `jawaban` (T49).

    Lampiran tanpa `disebut` selalu lolos. Bila ada dua atau lebih lampiran
    ber-`disebut`, hanya yang namanya tertulis di jawaban yang lolos. Nama yang
    lebih panjang dicocokkan lebih dulu lalu dihapus dari teks, sehingga "Basis
    Data Lanjut" di jawaban tidak ikut dihitung sebagai "Basis Data". Bila tidak
    satu nama pun tertulis, misalnya model hanya menulis "Kecerdasan Buatan"
    untuk "Artificial Intelligence", semuanya lolos: daftar yang berlebih lebih
    baik daripada jawaban "diampu 35 dosen" tanpa daftar yang dijanjikan
    `CATATAN_LAMPIRAN`."""
    calon = [i for i, a in enumerate(lampiran) if a.disebut]
    if len(calon) < 2:
        return list(lampiran)
    teks = kunci_sebutan(jawaban)
    tertulis: set[int] = set()
    for i in sorted(calon, key=lambda i: len(lampiran[i].disebut), reverse=True):
        pola = rf"(?<!\w){re.escape(kunci_sebutan(lampiran[i].disebut))}(?!\w)"
        teks, jumlah = re.subn(pola, " ", teks)
        if jumlah:
            tertulis.add(i)
    if not tertulis:
        return list(lampiran)
    return [a for i, a in enumerate(lampiran) if not a.disebut or i in tertulis]


@dataclass(frozen=True)
class ToolSpec:
    """Satu tool: skema untuk LLM, pemicu kelayakan, handler, label sitasi."""

    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema argumen (gaya OpenAI function)."""
    handler: Callable[..., Awaitable[ToolResult]]
    """Async, menerima argumen bernama yang sudah divalidasi, mengembalikan ToolResult."""
    triggers: tuple[str, ...] = ()
    """Kata kunci rute kelayakan (docs/tool-call.md §8). Kosong = tak pernah eligible.

    Pemicu tidak melihat unit pilihan mahasiswa: data SADS bersifat lintas-unit,
    dan kelayakan dihitung hanya dari teks pertanyaan (`ToolRegistry.eligible`)."""
    citation_label: str = ""

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def cocok(self, pertanyaan: str) -> bool:
        teks = pertanyaan.casefold()
        return any(pemicu.casefold() in teks for pemicu in self.triggers)


def validasi_argumen(spec: ToolSpec, mentah: dict[str, Any] | None) -> dict[str, Any]:
    """Tegakkan skema sebelum handler dipanggil (docs/tool-call.md §11).

    Argumen tak dikenal dibuang, string dibersihkan dari karakter non-cetak dan
    dibatasi panjangnya, dan argumen wajib yang kosong menolak panggilan. Handler
    tidak pernah menerima URL -- hanya nilai argumen bernama dari skema ini.
    """
    props = spec.parameters.get("properties", {})
    wajib = spec.parameters.get("required", [])
    bersih: dict[str, Any] = {}
    for nama, nilai in (mentah or {}).items():
        if nama not in props:
            continue
        if props[nama].get("type") == "string":
            # Model kadang mengisi argumen opsional dengan null alih-alih
            # menghilangkannya; itu berarti "tidak disaring", bukan "None".
            if nilai is None:
                continue
            teks = "".join(ch for ch in str(nilai) if ch.isprintable())
            bersih[nama] = teks[:MAKS_PANJANG_ARGUMEN].strip()
        else:
            bersih[nama] = nilai
    for nama in wajib:
        if not bersih.get(nama):
            raise ToolArgumentError(f"argumen wajib '{nama}' tidak ada atau kosong")
    return bersih


def hasil_tool_ke_dokumen(hasil: Iterable[ToolResult]) -> list[Document]:
    """Kartu sitasi sintetis: satu Document semu per label (docs/tool-call.md §10).

    Hasil tool bukan baris di tabel `documents`; Document semu ini hanya
    menumpang mesin sitasi yang ada (`citations_for`, `ringkas_sitasi_tanpa_halaman`).
    `jenis` `tanya_jawab` membuatnya diperlakukan sebagai sumber tak berhalaman
    di pipeline (penanda `[Judul]`). `dari_tool` membedakannya dari entri tanya
    jawab admin di kartu: `citations_for` memberinya `CitationOut.type` `data`,
    karena label "Tanya jawab resmi" keliru untuk data SADS (T44). Gabungan per
    label supaya "Data akademik SADS" menjadi satu kartu walau beberapa tool
    dipanggil.
    """
    per_label: dict[str, list[str]] = {}
    for r in hasil:
        if r.ok and r.text.strip():
            per_label.setdefault(r.label, []).append(r.text)
    return [
        Document(
            page_content="\n\n".join(teks),
            metadata={
                "judul": label,
                "jenis": DocumentType.TANYA_JAWAB,
                "halaman": 1,
                "file_path": "",
                "document_id": "",
                "dari_tool": True,
            },
        )
        for label, teks in per_label.items()
    ]
