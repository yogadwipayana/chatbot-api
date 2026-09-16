"""Fixture bersama.

Unit test tidak boleh butuh database maupun kunci API. Test yang butuh
Postgres ditandai `@pytest.mark.integration` dan dilewati otomatis bila
`TEST_DATABASE_URL` tidak diset.
"""

from __future__ import annotations

import os

import pytest

from tests.fixtures.fakes import (  # noqa: F401  (re-export sebagai fixture)
    FakeEmbeddings,
    FakeRetriever,
    RecordingLLM,
    RecordingRewriter,
    make_document,
)


@pytest.fixture
def llm() -> RecordingLLM:
    return RecordingLLM()


@pytest.fixture
def rewriter() -> RecordingRewriter:
    return RecordingRewriter()


@pytest.fixture
def strong_documents():
    """Hasil retrieval yang jelas relevan -- di atas ambang FR-3."""
    return [
        make_document("c1", halaman=12, vector_score=0.82),
        make_document("c2", halaman=13, vector_score=0.71),
    ]


@pytest.fixture
def weak_documents():
    """Hasil retrieval yang lemah -- harus memicu penolakan FR-3."""
    return [
        make_document("c1", halaman=4, vector_score=0.11, lexical_score=0.001),
        make_document("c2", halaman=5, vector_score=0.09, lexical_score=0.0),
    ]


@pytest.fixture
def strong_retriever(strong_documents) -> FakeRetriever:
    return FakeRetriever(strong_documents)


@pytest.fixture
def weak_retriever(weak_documents) -> FakeRetriever:
    return FakeRetriever(weak_documents)


@pytest.fixture
def empty_retriever() -> FakeRetriever:
    return FakeRetriever([])


def pytest_collection_modifyitems(config, items):
    """Lewati test integrasi bila tidak ada database uji."""
    if os.getenv("TEST_DATABASE_URL"):
        return
    skip = pytest.mark.skip(reason="TEST_DATABASE_URL tidak diset")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
