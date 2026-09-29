"""Struktur graf LangGraph (flow.md §4).

Perilaku tiap jalur sudah diuji di `test_pipeline.py` dan `test_gate.py` lewat
`run_pipeline`. Di sini yang dijaga adalah bentuk grafnya: urutan node dan
jalan keluar lebih awal yang dijanjikan flow.md tidak boleh bergeser diam-diam.
"""

from __future__ import annotations

from app.observability.applog import WRAPPER_NODES
from app.rag.graph import SEARCH_NODE, _search_graph, build_graph

URUTAN_AWAL = ["sanitize", "sensitive", "smalltalk", "rule_gate"]


def sisi() -> set[tuple[str, str]]:
    return {(e.source, e.target) for e in build_graph().get_graph().edges}


def test_seluruh_node_ada():
    nodes = set(build_graph().get_graph().nodes)
    assert nodes == set(URUTAN_AWAL) | {
        "jev_gate",
        SEARCH_NODE,
        "validate_context",
        "refuse",
        "generate",
        "__start__",
        "__end__",
    }


def test_urutan_awal():
    edges = sisi()
    assert ("__start__", "sanitize") in edges
    for asal, tujuan in zip(URUTAN_AWAL, URUTAN_AWAL[1:], strict=False):
        assert (asal, tujuan) in edges, f"{asal} -> {tujuan} hilang"


def test_gerbang_dan_pencarian_bercabang_dari_saringan_aturan():
    edges = sisi()
    assert ("rule_gate", "jev_gate") in edges
    assert ("rule_gate", SEARCH_NODE) in edges
    assert ("smalltalk", "jev_gate") not in edges


def test_keduanya_bertemu_di_validate_context():
    edges = sisi()
    assert ("jev_gate", "validate_context") in edges
    assert (SEARCH_NODE, "validate_context") in edges


def test_subgraph_pencarian_rewrite_lalu_retrieve():
    edges = {(e.source, e.target) for e in _search_graph().get_graph().edges}
    urutan = {("__start__", "rewrite"), ("rewrite", "retrieve"), ("retrieve", "__end__")}
    assert urutan <= edges


def test_jalan_keluar_lebih_awal():
    edges = sisi()
    for node in ("sensitive", "smalltalk", "rule_gate"):
        assert (node, "__end__") in edges
    # Vonis JEV baru ditindaklanjuti di titik temu, setelah pencarian selesai.
    assert ("validate_context", "__end__") in edges


def test_llm_hanya_dicapai_lewat_validate_context():
    edges = sisi()
    assert {asal for asal, tujuan in edges if tujuan == "generate"} == {"validate_context"}
    assert ("validate_context", "refuse") in edges
    assert ("refuse", "__end__") in edges


def test_pembungkus_pencarian_tidak_dicatat_sebagai_langkah():
    """Nama di `applog` dan di graf harus sama; bila bergeser, halaman Log
    mulai menghitung rewrite + retrieve dua kali dengan nama yang tak dikenal."""
    assert SEARCH_NODE in WRAPPER_NODES


def test_diagram_mermaid_dapat_dibuat():
    assert "jev_gate" in build_graph().get_graph().draw_mermaid()
