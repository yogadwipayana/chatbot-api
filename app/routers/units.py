"""Menu topik chatbot (unit layanan dan pertanyaan per unit), juga isian unit di dashboard."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from app.deps import UnitDirectoryDep, unit_terdaftar
from app.schemas.chat import FaqQuestion, UnitOut
from app.schemas.common import Error

router = APIRouter(prefix="/api", tags=["unit"])


@router.get("/units", response_model=list[UnitOut])
async def list_units(units: UnitDirectoryDep) -> list[UnitOut]:
    """Unit aktif, dalam urutan tampil di menu.

    Tidak tunduk pada kill switch: dashboard admin memakai daftar yang sama, dan
    admin justru bekerja saat layanan chat dimatikan.
    """
    return [UnitOut(nama=u.nama, deskripsi=u.deskripsi) for u in await units.list()]


@router.get(
    "/faq/questions", response_model=list[FaqQuestion], responses={422: {"model": Error}}
)
async def list_faq_questions(
    units: UnitDirectoryDep,
    unit: Annotated[
        str | None,
        Query(max_length=200, description="`nama` dari `GET /api/units`; kosong = semua."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=20)] = 6,
) -> list[FaqQuestion]:
    """Pertanyaan yang sering diajukan untuk satu topik, dari entri tanya jawab admin.

    Bukan saran karangan: setiap pertanyaan adalah entri yang diketik admin
    lengkap dengan jawabannya, jadi mengkliknya selalu terjawab. Topik tanpa
    entri mengembalikan daftar kosong, dan menu cukup mengajak mengetik.
    """
    resmi = await unit_terdaftar(units, unit) if unit and unit.strip() else None
    return [FaqQuestion(pertanyaan=p) for p in await units.pertanyaan(resmi, limit)]
