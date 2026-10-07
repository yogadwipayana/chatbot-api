"""Tool-calling: data layanan akademik yang diambil LLM saat menjawab.

Lihat `docs/tool-call.md`. Paket ini berisi primitif (`base`), klien HTTP
(`client`), handler layanan (`sads`), registry + rute kelayakan (`registry`),
dan loop agentik (`loop`). Semua di belakang sakelar `TOOLS_ENABLED`.
"""
