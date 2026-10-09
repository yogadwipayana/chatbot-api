"""Objek palsu untuk test.

Tidak ada test yang boleh memanggil API LLM, API embedding, atau LangSmith.
Selain soal biaya, test yang menyentuh jaringan tidak bisa membuktikan
invarian "LLM tidak dipanggil" -- yang justru inti FR-3 dan FR-7.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import bcrypt

from app.admin.accounts import EDITABLE_FIELDS, Account, DuplicateEmailError
from app.admin.permissions import ROLE_LEVEL, AdminRole
from app.admin.runtime_config import NilaiTersimpan
from app.embed_keys import EDITABLE_FIELDS as EMBED_EDITABLE_FIELDS
from app.embed_keys import KunciSematan, bentuk_sah, buat_kunci
from app.security.killswitch import KillSwitch
from app.units import (
    EDITABLE_FIELDS as UNIT_EDITABLE_FIELDS,
)
from app.units import (
    DuplicateUnitError,
    UnitInfo,
    UnitRecord,
    bentrok,
    cocokkan,
)


@dataclass
class StubDocument:
    """Pengganti `langchain_core.documents.Document` untuk unit test."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def make_document(
    chunk_id: str = "c1",
    *,
    judul: str = "Panduan Akademik 2025",
    halaman: int = 12,
    konten: str = "Isi chunk.",
    vector_score: float | None = 0.80,
    lexical_score: float | None = None,
    rrf_score: float = 0.016,
    document_id: str = "d1",
) -> StubDocument:
    raw: dict[str, float] = {}
    ranks: dict[str, int] = {}
    if vector_score is not None:
        raw["vector"] = vector_score
        ranks["vector"] = 1
    if lexical_score is not None:
        raw["fulltext"] = lexical_score
        ranks["fulltext"] = 1
    return StubDocument(
        page_content=konten,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "judul": judul,
            "halaman": halaman,
            "file_path": f"storage/documents/{document_id}.pdf",
            "rrf_score": rrf_score,
            "raw_scores": raw,
            "ranks": ranks,
        },
    )


class FakeRetriever:
    """Mengembalikan dokumen yang sudah ditentukan, mencatat query dan unit yang masuk."""

    def __init__(self, documents: Sequence[StubDocument] | None = None) -> None:
        self.documents = list(documents or [])
        self.queries: list[str] = []
        self.units: list[str | None] = []
        self.original_queries: list[str | None] = []
        """Pertanyaan asli yang ikut dicari bila `query` hasil rewrite (T40)."""

    async def ainvoke(
        self, query: str, *, unit: str | None = None, original_query: str | None = None
    ) -> list[StubDocument]:
        self.queries.append(query)
        self.units.append(unit)
        self.original_queries.append(original_query)
        return list(self.documents)


class RecordingLLM:
    """Mencatat setiap pemanggilan. `calls` kosong = LLM tidak pernah dipanggil."""

    def __init__(self, reply: str = "Jawaban [Panduan Akademik 2025, hal. 12].") -> None:
        self.reply = reply
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.run_id = "11111111-1111-4111-8111-111111111111"

    async def __call__(self, wrapped_question: str, documents) -> str:
        self.calls.append((wrapped_question, tuple(documents)))
        return self.reply

    async def stream(self, wrapped_question: str, documents):
        """Jalur streaming FE-1: jawaban yang sama, tiba sepotong demi sepotong.

        Sengaja dipecah per kata. Pengganti yang mengembalikan jawaban utuh
        dalam satu potongan tetap membuat test lulus padahal mahasiswa melihat
        jawabannya muncul sekaligus -- justru keluhan yang memicu fitur ini.
        """
        self.calls.append((wrapped_question, tuple(documents)))
        for indeks, kata in enumerate(self.reply.split(" ")):
            yield kata if indeks == 0 else f" {kata}"

    @property
    def called(self) -> bool:
        return bool(self.calls)

    @property
    def last_question(self) -> str:
        assert self.calls, "LLM belum pernah dipanggil"
        return self.calls[-1][0]


class RecordingRewriter:
    """Pengganti chain penulisan ulang query (FR-4)."""

    def __init__(self, rewritten: str = "pertanyaan mandiri hasil tulis ulang") -> None:
        self.rewritten = rewritten
        self.calls: list[tuple[str, str]] = []
        self.run_id = "22222222-2222-4222-8222-222222222222"

    async def __call__(self, question: str, history: str) -> str:
        self.calls.append((question, history))
        return self.rewritten

    @property
    def called(self) -> bool:
        return bool(self.calls)


class FakeEmbeddings:
    """Embedding deterministik tanpa jaringan; hanya untuk uji bentuk data."""

    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions
        self.queries: list[str] = []

    async def aembed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        base = float(sum(text.encode("utf-8")) % 97) / 97.0
        return [base] * self.dimensions

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [await self.aembed_query(t) for t in texts]


class FakeChatLogger:
    """Pengganti `ChatLogger`: menyimpan entri di memori, tanpa database."""

    def __init__(self, message_id: str = "9c3e1a44-6b2d-4f51-8a70-2d9b5c1e7f03") -> None:
        self.message_id = message_id
        self.fail = False
        self.entries: list[Any] = []

    async def log(self, entry) -> str | None:
        self.entries.append(entry)
        return None if self.fail else self.message_id


class FakeJawabanTerkirim:
    """Pengganti `SqlJawabanTerkirim`.

    Jawaban yang dicatat `FakeChatLogger` ikut terhitung, jadi alur tanya lalu
    tanya lanjutan di test berjalan seperti di produksi. `tambah` mengisi
    jawaban lama yang tidak lewat test itu sendiri.
    """

    def __init__(self, chat_logger: FakeChatLogger | None = None) -> None:
        self.chat_logger = chat_logger
        self.per_sesi: dict[str, list[str]] = {}
        self.gagal = False
        self.ditanya = 0

    def tambah(self, session_id: str, *jawaban: str) -> None:
        self.per_sesi.setdefault(session_id, []).extend(jawaban)

    async def terakhir(self, session_id: str, batas: int) -> list[str]:
        self.ditanya += 1
        if self.gagal:
            raise RuntimeError("database tidak dapat dihubungi")
        tercatat = list(self.per_sesi.get(session_id, []))
        if self.chat_logger is not None:
            tercatat += [
                e.outcome.text for e in self.chat_logger.entries if e.session_id == session_id
            ]
        return tercatat[::-1][:batas]


class FakeLogSink:
    """Pengganti `LogWriter`: baris log SQLite ditampung di memori."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, Any]]] = []

    def kirim(self, jenis: str, data: dict[str, Any]) -> None:
        self.rows.append((jenis, data))

    def of(self, jenis: str) -> list[dict[str, Any]]:
        return [data for j, data in self.rows if j == jenis]


class FakeAccountStore:
    """Pengganti `SqlAccountStore`: akun dashboard di memori, tanpa database."""

    def __init__(self) -> None:
        self.accounts: dict[uuid.UUID, Account] = {}
        self.logins: list[uuid.UUID] = []

    def add(
        self,
        email: str,
        role: AdminRole,
        *,
        password: str = "kata-sandi-admin-yang-panjang",
        unit: str | None = None,
        name: str | None = None,
        is_active: bool = True,
    ) -> Account:
        # rounds=4: bcrypt default sengaja lambat, dan fixture ini dibuat per test.
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=4)).decode()
        account = Account(
            id=uuid.uuid4(),
            email=email.lower(),
            role=AdminRole(role),
            password_hash=password_hash,
            is_active=is_active,
            unit=unit,
            name=name,
            created_at=datetime.now(UTC),
        )
        self.accounts[account.id] = account
        return account

    def by_email(self, email: str) -> Account:
        return next(a for a in self.accounts.values() if a.email == email.lower())

    def change(self, email: str, **changes: Any) -> Account:
        """Ubah akun langsung, seolah superadmin lain mengubahnya di tab berbeda."""
        account = replace(self.by_email(email), **changes)
        self.accounts[account.id] = account
        return account

    async def find_by_email(self, email: str) -> Account | None:
        return next(
            (a for a in self.accounts.values() if a.email == email.strip().lower()), None
        )

    async def get(self, account_id: uuid.UUID) -> Account | None:
        return self.accounts.get(account_id)

    async def list(self) -> list[Account]:
        return sorted(self.accounts.values(), key=lambda a: (-ROLE_LEVEL[a.role], a.email))

    async def create(
        self,
        *,
        email: str,
        name: str | None,
        role: AdminRole,
        unit: str | None,
        password_hash: str,
    ) -> Account:
        if await self.find_by_email(email):
            raise DuplicateEmailError(email)
        account = Account(
            id=uuid.uuid4(),
            email=email.lower(),
            role=AdminRole(role),
            password_hash=password_hash,
            name=name,
            unit=unit,
            created_at=datetime.now(UTC),
        )
        self.accounts[account.id] = account
        return account

    async def update(self, account_id: uuid.UUID, changes: dict[str, Any]) -> Account | None:
        account = self.accounts.get(account_id)
        if account is None:
            return None
        data = {k: v for k, v in changes.items() if k in EDITABLE_FIELDS}
        if "role" in data:
            data["role"] = AdminRole(data["role"])
        account = replace(account, **data)
        self.accounts[account_id] = account
        return account

    async def set_password(
        self, account_id: uuid.UUID, password_hash: str, changed_at: datetime
    ) -> None:
        self.accounts[account_id] = replace(
            self.accounts[account_id],
            password_hash=password_hash,
            password_changed_at=changed_at,
        )

    async def delete(self, account_id: uuid.UUID) -> bool:
        return self.accounts.pop(account_id, None) is not None

    async def count_active_superadmins(self) -> int:
        return sum(
            1 for a in self.accounts.values() if a.role is AdminRole.SUPERADMIN and a.is_active
        )

    async def record_login(self, account_id: uuid.UUID) -> None:
        self.logins.append(account_id)
        self.accounts[account_id] = replace(
            self.accounts[account_id], last_login_at=datetime.now(UTC)
        )


class FakeRuntimeConfigStore:
    """Pengganti `SqlRuntimeConfigStore`: penimpaan setelan di memori."""

    def __init__(self, awal: dict[str, str] | None = None) -> None:
        self.values: dict[str, NilaiTersimpan] = {
            k: NilaiTersimpan(v, datetime.now(UTC), "seed@instiki.ac.id")
            for k, v in (awal or {}).items()
        }

    async def load(self) -> dict[str, NilaiTersimpan]:
        return dict(self.values)

    async def replace(self, changes, *, by: str) -> None:
        for key, value in changes.items():
            if value is None:
                self.values.pop(key, None)
            else:
                self.values[key] = NilaiTersimpan(value, datetime.now(UTC), by)

    async def clear(self) -> None:
        self.values.clear()


class FakeKillSwitchStore:
    """Pengganti `SqlKillSwitchStore`: tabel `kill_switch` di memori.

    `gagal=True` meniru database yang menolak tulis (mis. tabel belum dimigrasi).
    """

    def __init__(self, tersimpan: KillSwitch | None = None, *, gagal: bool = False) -> None:
        self.tersimpan = tersimpan
        self.gagal = gagal
        self.simpan_ke = 0

    async def load(self) -> KillSwitch | None:
        if self.gagal:
            raise RuntimeError('relation "kill_switch" does not exist')
        return replace(self.tersimpan) if self.tersimpan else None

    async def save(self, switch: KillSwitch) -> None:
        if self.gagal:
            raise RuntimeError('relation "kill_switch" does not exist')
        self.simpan_ke += 1
        self.tersimpan = replace(switch) if switch.engaged else None


UNIT_RESMI = (
    "BAAK",
    "FO",
    "Keuangan",
    "Kemahasiswaan",
    "Prodi",
    "Fakultas",
    "PLK",
    "UPS",
    "Akademik",
)
"""Sama dengan isi awal migrasi 0009."""


def _urutan(unit: UnitRecord) -> tuple[int, str]:
    return (unit.sort_order, unit.name)


class FakeUnitDirectory:
    """Pengganti `SqlUnitDirectory`: daftar unit di memori, aturan cocok yang sama."""

    def __init__(self, names: Sequence[str] = UNIT_RESMI) -> None:
        self.records = [UnitRecord(n, None, i, True) for i, n in enumerate(names, start=1)]
        self.faq: dict[str, list[str]] = {}
        """unit -> pertanyaan entri tanya jawab, terbaru lebih dulu."""
        self.diminta: list[tuple[str | None, int]] = []

    @property
    def units(self) -> list[UnitInfo]:
        """Unit aktif, dalam urutan menu -- seperti `SqlUnitDirectory.list`."""
        aktif = [u for u in sorted(self.records, key=_urutan) if u.is_active]
        return [UnitInfo(u.name, u.description) for u in aktif]

    async def list(self) -> list[UnitInfo]:
        return list(self.units)

    async def pertanyaan(self, unit: str | None, limit: int) -> list[str]:
        self.diminta.append((unit, limit))
        semua = self.faq.get(unit, []) if unit else [p for ps in self.faq.values() for p in ps]
        return semua[:limit]

    async def resolve(self, name: str | None) -> str | None:
        return cocokkan(self.units, name)

    async def semua(self) -> list[UnitRecord]:
        return sorted(self.records, key=_urutan)

    async def ambil(self, name: str) -> UnitRecord | None:
        return next((u for u in self.records if u.name == name), None)

    async def buat(
        self, *, name: str, description: str | None, sort_order: int | None
    ) -> UnitRecord:
        if bentrok(self.records, name):
            raise DuplicateUnitError(name)
        if sort_order is None:
            sort_order = max((u.sort_order for u in self.records), default=0) + 1
        unit = UnitRecord(name, description, sort_order, True)
        self.records.append(unit)
        return unit

    async def ubah(self, name: str, changes: dict[str, Any]) -> UnitRecord | None:
        lama = await self.ambil(name)
        if lama is None:
            return None
        baru = changes.get("name", name)
        if baru != name and bentrok(self.records, baru, kecuali=name):
            raise DuplicateUnitError(baru)
        unit = replace(lama, **{k: v for k, v in changes.items() if k in UNIT_EDITABLE_FIELDS})
        self.records[self.records.index(lama)] = unit
        return unit


class FakeEmbedKeyStore:
    """Pengganti `SqlEmbedKeyStore`: kunci sematan di memori."""

    def __init__(self) -> None:
        self.keys: dict[str, KunciSematan] = {}

    def add(
        self,
        name: str = "Situs PMB",
        allowed_origins: Sequence[str] = (),
        *,
        is_active: bool = True,
        questions_30d: int = 0,
    ) -> KunciSematan:
        """Penyiapan data test, tanpa melewati endpoint."""
        kunci = KunciSematan(
            key=buat_kunci(),
            name=name,
            allowed_origins=list(allowed_origins),
            is_active=is_active,
            created_by="uji@instiki.ac.id",
            created_at=datetime.now(UTC),
            questions_30d=questions_30d,
        )
        self.keys[kunci.key] = kunci
        return kunci

    async def aktif(self, kunci: str) -> KunciSematan | None:
        hasil = self.keys.get(kunci) if bentuk_sah(kunci) else None
        return hasil if hasil is not None and hasil.is_active else None

    async def semua(self) -> list[KunciSematan]:
        # Sort stabil: jam Windows cukup kasar untuk memberi dua kunci created_at
        # yang sama, dan urutan pembuatan yang menjadi penentu seri.
        return sorted(self.keys.values(), key=lambda k: k.created_at)

    async def ambil(self, kunci: str) -> KunciSematan | None:
        return self.keys.get(kunci)

    async def buat(self, *, name: str, allowed_origins: list[str], oleh: str) -> KunciSematan:
        kunci = self.add(name, allowed_origins)
        kunci = replace(kunci, created_by=oleh)
        self.keys[kunci.key] = kunci
        return kunci

    async def ubah(self, kunci: str, changes: dict[str, Any]) -> KunciSematan | None:
        lama = self.keys.get(kunci)
        if lama is None:
            return None
        baru = replace(
            lama, **{k: v for k, v in changes.items() if k in EMBED_EDITABLE_FIELDS}
        )
        self.keys[kunci] = baru
        return baru

    async def hapus(self, kunci: str) -> bool:
        return self.keys.pop(kunci, None) is not None
