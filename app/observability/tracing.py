"""LangSmith tracing (FR-8).

Aktif di semua environment. PRD §12 menyebut "debugging tersembunyi di balik
abstraksi" sebagai risiko; tracing adalah mitigasinya, jadi ia bukan opsional.
"""

from __future__ import annotations

import os

from app.config import Settings


def configure_tracing(settings: Settings) -> bool:
    """Set variabel lingkungan yang dibaca LangSmith. Return: aktif atau tidak."""
    if not settings.langsmith_tracing or settings.langsmith_api_key is None:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    return True
