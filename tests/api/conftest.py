"""Fixture untuk test lapisan API.

Seluruh ketergantungan luar diganti lewat `dependency_overrides`: tidak ada
koneksi database, tidak ada panggilan API LLM maupun embedding.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.admin.permissions import AdminRole
from app.config import get_settings
from app.deps import (
    build_llm_call,
    build_retriever,
    get_account_store,
    get_chat_logger,
    get_embeddings,
    get_runtime_config_store,
    get_session,
    get_storage,
)
from app.main import create_app
from app.security.auth import create_access_token
from app.security.killswitch import KillSwitch, get_kill_switch
from app.security.ratelimit import (
    LOGIN_MAX_FAILURES,
    LOGIN_WINDOW_SECONDS,
    FailureLimiter,
    get_login_limiter,
)
from tests.fixtures.fakes import (
    FakeAccountStore,
    FakeChatLogger,
    FakeEmbeddings,
    FakeRetriever,
    FakeRuntimeConfigStore,
    RecordingLLM,
)

SANDI = "kata-sandi-admin-yang-panjang"

ADMIN_EMAIL = "admin@kampus.ac.id"
"""Superadmin: test lama yang tidak peduli level memakai akun ini."""
ADMIN_BIASA_EMAIL = "admin.biasa@kampus.ac.id"
STAF_EMAIL = "staf.keuangan@kampus.ac.id"
STAF_UNIT = "Biro Keuangan"

EMAIL_PER_LEVEL = {
    AdminRole.STAF: STAF_EMAIL,
    AdminRole.ADMIN: ADMIN_BIASA_EMAIL,
    AdminRole.SUPERADMIN: ADMIN_EMAIL,
}


@pytest.fixture
def kill_switch() -> KillSwitch:
    return KillSwitch()


@pytest.fixture
def api_llm() -> RecordingLLM:
    return RecordingLLM()


@pytest.fixture
def chat_logger() -> FakeChatLogger:
    return FakeChatLogger()


@pytest.fixture
def login_limiter() -> FailureLimiter:
    """Baru per test, supaya kegagalan login satu test tidak mengunci test lain."""
    return FailureLimiter(LOGIN_MAX_FAILURES, LOGIN_WINDOW_SECONDS)


@pytest.fixture
def runtime_config() -> FakeRuntimeConfigStore:
    """Tanpa penimpaan: setiap test berangkat dari nilai `.env`."""
    return FakeRuntimeConfigStore()


@pytest.fixture
def accounts() -> FakeAccountStore:
    """Satu akun untuk setiap level, semuanya dengan kata sandi `SANDI`."""
    store = FakeAccountStore()
    store.add(ADMIN_EMAIL, AdminRole.SUPERADMIN, password=SANDI, nama="Super Admin")
    store.add(ADMIN_BIASA_EMAIL, AdminRole.ADMIN, password=SANDI)
    store.add(STAF_EMAIL, AdminRole.STAF, password=SANDI, unit=STAF_UNIT)
    return store


@pytest.fixture
def make_client(kill_switch, api_llm, chat_logger, login_limiter, accounts, runtime_config):
    """Bangun TestClient dengan retriever yang hasilnya ditentukan test."""

    def factory(documents, *, session=None) -> TestClient:
        app = create_app()
        app.dependency_overrides[build_retriever] = lambda: FakeRetriever(documents)
        app.dependency_overrides[build_llm_call] = lambda: api_llm
        app.dependency_overrides[get_kill_switch] = lambda: kill_switch
        app.dependency_overrides[get_session] = lambda: session
        app.dependency_overrides[get_chat_logger] = lambda: chat_logger
        app.dependency_overrides[get_login_limiter] = lambda: login_limiter
        app.dependency_overrides[get_account_store] = lambda: accounts
        app.dependency_overrides[get_runtime_config_store] = lambda: runtime_config
        app.dependency_overrides[get_embeddings] = lambda: FakeEmbeddings()
        app.dependency_overrides[get_storage] = lambda: None
        return TestClient(app)

    return factory


@pytest.fixture
def client(make_client, strong_documents) -> TestClient:
    return make_client(strong_documents)


@pytest.fixture
def payload() -> dict:
    return {"question": "kapan pengisian KRS dibuka?", "session_id": "sesi-uji-12345"}


@pytest.fixture
def headers_for():
    """Header bearer untuk akun dengan email tertentu."""
    secret = get_settings().admin_jwt_secret.get_secret_value()

    def build(email: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {create_access_token(email, secret)}"}

    return build


@pytest.fixture
def admin_headers(headers_for) -> dict[str, str]:
    return headers_for(ADMIN_EMAIL)
