"""Struktur graf LangGraph (flow.md §4).

Perilaku tiap jalur sudah diuji di `test_pipeline.py` dan `test_gate.py` lewat
`run_pipeline`. Di sini yang dijaga adalah bentuk grafnya: urutan node dan
jalan keluar lebih awal yang dijanjikan flow.md tidak boleh bergeser diam-diam.
"""

from __future__ import annotations

from app.rag.graph import build_graph

URUTAN = [
    "sanitize",
    "sensitive",
    "smalltalk",
    "jev_gate",
    "rewrite",
    "retrieve",
    "validate_context",
]


def sisi() -> set[tuple[str, str]]:
    return {(e.source, e.target) for e in build_graph().get_graph().edges}


def test_seluruh_node_ada():
    nodes = set(build_graph().get_graph().nodes)
    assert set(URUTAN) | {"refuse", "generate", "__start__", "__end__"} == nodes


def test_urutan_utama():
    edges = sisi()
    assert ("__start__", "sanitize") in edges
    for asal, tujuan in zip(URUTAN, URUTAN[1:], strict=False):
        assert (asal, tujuan) in edges, f"{asal} -> {tujuan} hilang"


def test_jalan_keluar_lebih_awal_sebelum_retrieval():
    edges = sisi()
    for node in ("sensitive", "smalltalk", "jev_gate"):
        assert (node, "__end__") in edges


def test_llm_hanya_dicapai_lewat_validate_context():
    edges = sisi()
    assert {asal for asal, tujuan in edges if tujuan == "generate"} == {"validate_context"}
    assert ("validate_context", "refuse") in edges
    assert ("refuse", "__end__") in edges


def test_diagram_mermaid_dapat_dibuat():
    assert "jev_gate" in build_graph().get_graph().draw_mermaid()
