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
    """Mengembalikan dokumen yang sudah ditentukan, mencatat query yang masuk."""

    def __init__(self, documents: Sequence[StubDocument] | None = None) -> None:
        self.documents = list(documents or [])
        self.queries: list[str] = []

    async def ainvoke(self, query: str) -> list[StubDocument]:
        self.queries.append(query)
        return list(self.documents)


class RecordingLLM:
    """Mencatat setiap pemanggilan. `calls` kosong = LLM tidak pernah dipanggil."""

    def __init__(self, reply: str = "Jawaban [Panduan Akademik 2025, hal. 12].") -> None:
        self.reply = reply
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

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
        nama: str | None = None,
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
            nama=nama,
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
        nama: str | None,
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
            nama=nama,
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
            k: NilaiTersimpan(v, datetime.now(UTC), "seed@kampus.ac.id")
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
