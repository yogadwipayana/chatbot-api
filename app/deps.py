"""Dependency injection FastAPI.

Semua ketergantungan berat (DB, LLM, retriever) melewati fungsi di sini,
supaya test dapat menggantinya lewat `app.dependency_overrides` tanpa
menyentuh jaringan.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.permissions import ROLE_LABELS, AdminRole, CurrentAdmin
from app.config import Settings, get_settings
from app.db.session import get_session
from app.security.auth import decode_access_token
from app.security.killswitch import KillSwitch, get_kill_switch
from app.security.ratelimit import FailureLimiter, get_login_limiter

SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
KillSwitchDep = Annotated[KillSwitch, Depends(get_kill_switch)]
LoginLimiterDep = Annotated[FailureLimiter, Depends(get_login_limiter)]


def guard_kill_switch(switch: KillSwitchDep) -> None:
    """Blokir endpoint mahasiswa saat layanan dimatikan (FR-9)."""
    if switch.engaged:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=switch.message,
        )


@lru_cache(maxsize=1)
def _storage_singleton() -> Any:
    """Satu instans penyimpanan per proses.

    Klien boto3 relatif mahal dibuat (memuat data model service dan menyiapkan
    connection pool). Membuatnya ulang per permintaan akan terasa saat musim
    KRS -- PRD §11 memperkirakan beban puncak 3x lipat.

    Tidak menerima `Settings` sebagai argumen karena model Pydantic tidak
    hashable sehingga tidak bisa jadi kunci cache; `get_settings()` sendiri
    sudah singleton, jadi membacanya di sini setara.
    """
    from app.storage import build_storage

    return build_storage(get_settings())


def get_storage() -> Any:
    """Dependency penyimpanan objek. Di-override di test dengan penyimpanan lokal."""
    return _storage_singleton()


StorageDep = Annotated[Any, Depends(get_storage)]


def build_retriever(settings: SettingsDep) -> Any:
    """Rakit `PostgresHybridRetriever`. Di-override di test dengan retriever palsu.

    Retriever menerima pabrik sesi, bukan satu sesi: vector search dan fulltext
    search berjalan paralel dan masing-masing butuh koneksinya sendiri.
    """
    from app.db.session import SessionLocal
    from app.rag.providers import build_embeddings
    from app.rag.retriever import PostgresHybridRetriever

    embeddings = build_embeddings(settings)
    return PostgresHybridRetriever(
        session_factory=SessionLocal,
        embed_query=embeddings.aembed_query,
        candidates=settings.retrieval_candidates,
        top_n=settings.retrieval_top_n,
        rrf_k=settings.rrf_k,
        weight_vector=settings.rrf_weight_vector,
        weight_fulltext=settings.rrf_weight_fulltext,
    )


class LLMCall:
    """(pertanyaan_terbungkus, dokumen) -> teks jawaban, sambil mencatat pemakaian token.

    FastAPI membangun dependency ulang untuk setiap permintaan, jadi satu
    instans hanya melayani satu permintaan dan `usage` tidak tertukar antar
    permintaan yang berjalan bersamaan. Pemakaian token dibutuhkan estimasi
    biaya AD-5 (FR-8).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.chat_model
        self.usage: dict[str, Any] | None = None

    async def __call__(self, wrapped_question: str, documents) -> str:
        from app.rag.prompts import answer_prompt, format_context
        from app.rag.providers import build_llm

        chain = answer_prompt() | build_llm(self.settings, streaming=False)
        result = await chain.ainvoke(
            {"context": format_context(documents), "question": wrapped_question}
        )
        self.usage = getattr(result, "usage_metadata", None)
        return result.content


def build_llm_call(settings: SettingsDep) -> Any:
    """Kembalikan callable (pertanyaan_terbungkus, dokumen) -> teks jawaban.

    Disuntikkan sebagai dependency, bukan dibuat di dalam handler, supaya test
    API dapat menggantinya dan membuktikan LLM tidak dipanggil pada jalur
    penolakan (FR-3) dan pertanyaan sensitif (FR-7).
    """
    return LLMCall(settings)


def get_embeddings(settings: SettingsDep) -> Any:
    """Model embedding untuk ingestion (AD-3). Di-override di test dengan embedding palsu."""
    from app.rag.providers import build_embeddings

    return build_embeddings(settings)


def get_chat_logger() -> Any:
    """Pencatat percakapan ke Postgres (FR-8). Di-override di test."""
    from app.db.session import SessionLocal
    from app.observability.chatlog import ChatLogger

    return ChatLogger(SessionLocal)


_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="bearerAuth",
    bearerFormat="JWT",
    description="Token dari `POST /api/admin/login`. Hanya untuk endpoint admin.",
)


def get_account_store(session: SessionDep) -> Any:
    """Penyimpanan akun dashboard. Di-override di test dengan penyimpanan di memori."""
    from app.admin.accounts import SqlAccountStore

    return SqlAccountStore(session)


AccountStoreDep = Annotated[Any, Depends(get_account_store)]


async def require_admin(
    settings: SettingsDep,
    store: AccountStoreDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> CurrentAdmin:
    """Verifikasi bearer token admin (AD-1) dan muat akunnya dari database.

    Level dan unit diambil dari database, bukan dari isi token: menurunkan
    level, memindah unit, atau menonaktifkan akun berlaku pada permintaan
    berikutnya. Token yang terbit sebelum kata sandi terakhir diganti ditolak,
    sehingga mengatur ulang kata sandi juga mengeluarkan sesi yang bocor.

    Pesannya ditujukan ke admin non-teknis (PRD §9): dashboard menampilkannya
    apa adanya sebelum mengarahkan kembali ke halaman login.
    """
    if credentials is None or not credentials.credentials:
        raise _tidak_berwenang("Silakan masuk terlebih dahulu.")
    try:
        payload = decode_access_token(
            credentials.credentials, settings.admin_jwt_secret.get_secret_value()
        )
    except Exception as exc:  # jwt.PyJWTError dan turunannya
        raise _tidak_berwenang("Sesi Anda sudah berakhir. Silakan masuk kembali.") from exc

    account = await store.find_by_email(str(payload.get("sub", "")))
    if account is None or not account.is_active:
        raise _tidak_berwenang("Akun Anda sudah tidak aktif. Hubungi superadmin.")

    diganti = account.password_changed_at
    # `iat` dibulatkan ke detik; bandingkan dalam detik juga, supaya token yang
    # terbit pada detik yang sama dengan penggantian (mis. dari endpoint ganti
    # kata sandi itu sendiri) tidak langsung ditolak.
    if diganti is not None and int(payload.get("iat", 0)) < int(diganti.timestamp()):
        raise _tidak_berwenang("Kata sandi akun ini sudah diganti. Silakan masuk kembali.")

    return account.as_current()


def _tidak_berwenang(pesan: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, pesan, headers={"WWW-Authenticate": "Bearer"}
    )


CurrentAdminDep = Annotated[CurrentAdmin, Depends(require_admin)]

_PESAN_TERLARANG = {
    AdminRole.STAF: "Silakan masuk terlebih dahulu.",
    AdminRole.ADMIN: "Fitur ini hanya untuk Admin dan Superadmin.",
    AdminRole.SUPERADMIN: "Fitur ini hanya untuk Superadmin.",
}


def require_role(minimum: AdminRole):
    """Dependency: akun harus minimal berlevel `minimum` (403 bila tidak).

    Setiap operasi admin di `api.yaml` mencatat level minimumnya sebagai
    `x-min-role`; `tests/api/test_admin_roles.py` memastikan kode menegakkannya.
    """

    def check(admin: CurrentAdminDep) -> CurrentAdmin:
        if not admin.at_least(minimum):
            raise HTTPException(status.HTTP_403_FORBIDDEN, _PESAN_TERLARANG[minimum])
        return admin

    check.__name__ = f"require_{minimum.value}"
    check.__doc__ = f"Minimal {ROLE_LABELS[minimum]}."
    return check
