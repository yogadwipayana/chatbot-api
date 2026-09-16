"""Filter dokumen aktif (FR-2).

PRD §6 menolak metadata filter bawaan VectorStore: penyaringan ini harus
terjadi di level SQL, di dalam WHERE, supaya dokumen kedaluwarsa tidak pernah
ikut terambil lalu disaring belakangan. Metrik §3 menargetkan 0 dokumen
kedaluwarsa aktif di indeks.

Dipisahkan ke modul sendiri agar satu-satunya definisi predikat ini dapat
diuji tanpa database, dan agar tidak ada query retrieval yang lupa memakainya.
"""

from __future__ import annotations

ACTIVE_DOCUMENT_PREDICATE = (
    "d.is_active = true AND (d.valid_until IS NULL OR d.valid_until > now())"
)
"""Wajib ada di SETIAP query retrieval. Alias tabel `documents` = `d`."""


def active_document_clause(alias: str = "d") -> str:
    """Bentuk predikat dokumen aktif untuk alias tabel tertentu."""
    if not alias.isidentifier():
        raise ValueError(f"alias tabel tidak valid: {alias!r}")
    return (
        f"{alias}.is_active = true "
        f"AND ({alias}.valid_until IS NULL OR {alias}.valid_until > now())"
    )
