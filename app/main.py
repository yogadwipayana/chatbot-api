"""Entrypoint FastAPI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.observability.applog import hentikan_log, mulai_log
from app.observability.tracing import configure_tracing
from app.routers import (
    admin_auth,
    admin_config,
    admin_documents,
    admin_embed_keys,
    admin_faq,
    admin_logs,
    admin_ops,
    admin_quality,
    admin_units,
    admin_users,
    chat,
    documents,
    embed,
    health,
    units,
)
from app.security.batas_body import BatasUkuranBody
from app.security.killswitch import KillSwitch, get_kill_switch, pulihkan
from app.security.sanitize import InvalidQuestion

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Paling awal: tanpa ini logger `app.*` belum punya handler, dan baris INFO
    # di bawah hilang sebelum sempat tercatat ke mana pun.
    mulai_log(settings)
    aktif = configure_tracing(settings)
    # Dicatat saat start, bukan hanya dibaca lewat /health: tracing yang mati
    # tidak menimbulkan galat apa pun, dan tanpa satu baris di log start tidak
    # ada momen lain yang memaksa siapa pun menyadarinya (FR-8).
    logger.info("Tracing LangSmith %s", "aktif" if aktif else "mati")
    await pulihkan_kill_switch(get_kill_switch())
    apply_initial_kill_switch(settings, get_kill_switch())
    try:
        yield
    finally:
        # Tulis sisa antrean log sebelum proses keluar.
        hentikan_log()


async def pulihkan_kill_switch(switch: KillSwitch, buat_store=None) -> None:
    """Kill switch yang menyala sebelum proses berhenti tetap menyala.

    Gagal membaca (DB mati, tabel `kill_switch` belum dimigrasi) tidak
    menghentikan start: layanan mulai dalam keadaan hidup, seperti sebelum
    status ini disimpan, dan kegagalannya tercatat di Log aplikasi.
    `buat_store` diganti di test; bawaannya tabel `kill_switch` lewat sesi baru.
    """
    try:
        if buat_store is None:
            from app.db.session import SessionLocal
            from app.security.killswitch import SqlKillSwitchStore

            async with SessionLocal() as session:
                dipulihkan = await pulihkan(switch, SqlKillSwitchStore(session))
        else:
            dipulihkan = await pulihkan(switch, buat_store())
    except Exception:
        logger.exception(
            "Status kill switch tersimpan tidak dapat dibaca; layanan chat mulai hidup"
        )
        return
    if dipulihkan:
        logger.warning(
            "Kill switch dipulihkan dari database: layanan chat tetap mati (oleh %s): %s",
            switch.engaged_by,
            switch.reason,
        )


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
    # Didaftarkan SEBELUM CORS: middleware terakhir menjadi lapisan terluar,
    # jadi CORS tetap membungkus 413 dari sini dan dashboard membaca pesannya,
    # bukan galat jaringan karena header CORS hilang.
    app.add_middleware(BatasUkuranBody, max_upload_mb=settings.max_upload_mb)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        # X-Embed-Key: panel `/embed` portal yang dimuat situs lain.
        allow_headers=["Authorization", "Content-Type", "X-Session-Id", "X-Embed-Key"],
        expose_headers=["Retry-After"],
    )
    app.add_exception_handler(InvalidQuestion, invalid_question_handler)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(units.router)
    app.include_router(embed.router)
    app.include_router(documents.router)
    app.include_router(admin_auth.router)
    app.include_router(admin_documents.router)
    app.include_router(admin_faq.router)
    app.include_router(admin_quality.router)
    app.include_router(admin_ops.router)
    app.include_router(admin_logs.router)
    app.include_router(admin_config.router)
    app.include_router(admin_users.router)
    app.include_router(admin_units.router)
    app.include_router(admin_embed_keys.router)
    return app


app = create_app()
