"""Tahap pertama evaluasi RAGAS: pemuat set evaluasi dan bentuk rekamannya."""

from __future__ import annotations

import json

import pytest

from app.rag.chain import OutcomeKind, PipelineOutcome
from app.rag.gate import GateLabel, GateVerdict
from app.rag.threshold import Decision, Reason, ThresholdDecision
from eval.run_generation import GenerationCase, build_record, load_cases, ringkas
from tests.fixtures.fakes import make_document


def tulis(tmp_path, *baris: str):
    path = tmp_path / "set.jsonl"
    path.write_text("\n".join(baris), encoding="utf-8")
    return path


class TestLoadCases:
    def test_komentar_dan_baris_kosong_dilewati(self, tmp_path):
        path = tulis(
            tmp_path,
            "// contoh",
            "",
            json.dumps({"question": "syarat cuti?", "reference": "dua semester"}),
            json.dumps({"question": "KRS?", "reference": "Agustus", "unit": "Akademik"}),
        )
        assert load_cases(path) == [
            GenerationCase("syarat cuti?", "dua semester"),
            GenerationCase("KRS?", "Agustus", "Akademik"),
        ]

    @pytest.mark.parametrize(
        "baris",
        [
            json.dumps({"question": "tanpa rujukan"}),
            json.dumps({"question": "rujukan kosong", "reference": "  "}),
            "{bukan json",
        ],
    )
    def test_baris_cacat_gagal_keras(self, tmp_path, baris):
        """Rujukan kosong membuat RAGAS menilai terhadap ketiadaan -- angkanya palsu."""
        with pytest.raises(ValueError, match=":1 tidak valid"):
            load_cases(tulis(tmp_path, baris))

    def test_berkas_kosong_ditolak(self, tmp_path):
        with pytest.raises(ValueError, match="tidak berisi"):
            load_cases(tulis(tmp_path, "// hanya komentar"))


class TestBuildRecord:
    def test_kolom_mengikuti_single_turn_sample(self):
        docs = (make_document("c1", konten="Pasal 12 ..."),)
        outcome = PipelineOutcome(
            kind=OutcomeKind.ANSWER,
            text="Cuti dapat diajukan ...",
            documents=docs,
            decision=ThresholdDecision(Decision.PROCEED, Reason.OK, 0.8, None, 0.91),
            gate=GateVerdict(GateLabel.ACADEMIC, 1.0, blocked=False),
            llm_called=True,
        )
        rekaman = build_record(GenerationCase("syarat cuti?", "dua semester"), outcome, 1200)
        assert rekaman["user_input"] == "syarat cuti?"
        assert rekaman["reference"] == "dua semester"
        assert rekaman["response"] == "Cuti dapat diajukan ..."
        assert rekaman["retrieved_contexts"] == ["Pasal 12 ..."]
        assert rekaman["kind"] == "answer"
        assert rekaman["top_rerank_score"] == 0.91
        assert rekaman["gate_label"] == "academic"
        json.dumps(rekaman)

    def test_ringkasan_memisahkan_yang_tidak_dijawab(self):
        hasil = [{"kind": "answer"}, {"kind": "refusal"}, {"kind": "rejected"}]
        assert ringkas(hasil).startswith("3 pertanyaan: 1 dijawab, 2 tidak")
