"""Entrypoint FastAPI."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.observability.tracing import configure_tracing
from app.routers import (
    admin_auth,
    admin_documents,
    admin_ops,
    admin_quality,
    admin_users,
    chat,
    documents,
    health,
)
from app.security.killswitch import KillSwitch, get_kill_switch
from app.security.sanitize import InvalidQuestion


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.tracing_enabled = configure_tracing(settings)
    apply_initial_kill_switch(settings, get_kill_switch())
    yield


def apply_initial_kill_switch(settings: Settings, switch: KillSwitch) -> None:
    """`KILL_SWITCH_ENABLED=true` menyalakan kill switch sejak proses mulai.

    Jalur cadangan untuk saat dashboard admin sendiri tidak dapat diakses:
    ubah variabel, restart kontainer.
    """
    if settings.kill_switch_enabled and not switch.engaged:
        switch.engage(
            "KILL_SWITCH_ENABLED=true saat layanan dijalankan", by="konfigurasi server"
        )


async def invalid_question_handler(request: Request, exc: InvalidQuestion) -> JSONResponse:
    """Pertanyaan yang kosong setelah disanitasi (mis. hanya spasi) lolos
    `min_length=1` Pydantic. Tanpa handler ini ia menjadi 500, bukan 422."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": [{"loc": ["body", "question"], "msg": str(exc), "type": "value_error"}]
        },
    )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Chatbot Administrasi Mahasiswa",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Session-Id"],
        expose_headers=["Retry-After"],
    )
    app.add_exception_handler(InvalidQuestion, invalid_question_handler)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(admin_auth.router)
    app.include_router(admin_documents.router)
    app.include_router(admin_quality.router)
    app.include_router(admin_ops.router)
    app.include_router(admin_users.router)
    return app


app = create_app()
