import tempfile
from pathlib import Path
import pytest

from aqp.schemas import (
    Anchor,
    Claim,
    EditingMetrics,
    Evaluation,
    EvidenceRecord,
    EvidenceStatus,
    FailureCode,
    RunStatus,
)
from aqp.editmetrics import calculate_edit_ratio, get_detailed_edit_report
from aqp.storage import StorageManager
from aqp.pipeline import EvaluationPipeline, PipelineInputError


def test_edit_ratio_identical_and_modified():
    """Kiểm thử DoD editmetrics: giống hệt -> 0%; xóa hết hoặc sửa đổi -> đúng công thức (4)."""
    text = "Cam sành Vĩnh Long thu hoạch ngày 20/10 đạt độ ngọt Brix 12%."
    m_same = calculate_edit_ratio(text, text, active_review_seconds=15.4)
    assert m_same.edit_ratio == 0.0
    assert m_same.active_review_seconds == 15.4

    # Thay 1 từ trong câu 4 từ -> khoảng cách cấp từ = 1, mẫu số = 4 -> 25.0%
    m_diff = calculate_edit_ratio("Thu hoạch ngày 15/10", "Thu hoạch ngày 20/10", 30.0)
    assert m_diff.edit_ratio == 25.0

    # Xóa hết -> mẫu số max(1, 0) = 1, khoảng cách = 4 -> 400.0%
    m_empty = calculate_edit_ratio("Thu hoạch ngày 15/10", "", 10.0)
    assert m_empty.edit_ratio == 400.0


def test_storage_roundtrip_and_immutability():
    """Kiểm thử DoD storage: Round-trip JSON không mất claim/anchor và bản gốc bất biến (NFR7)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = str(Path(tmpdir) / "test_aqp.db")
        store = StorageManager(db_path=db_file)

        evidence = [
            EvidenceRecord(source_id="SRC-001", record_id="REC-001", text="Ngày 20/10: thu hoạch lô A.")
        ]
        meta1 = store.save_initial_case("CASE-001", "SESSION-001", "Bài gốc ban đầu", evidence)
        # Gọi lại với nội dung khác cho cùng CASE-001 -> không được phép ghi đè bản gốc
        store.save_initial_case("CASE-001", "SESSION-001", "Bài cố tình ghi đè", evidence)

        v2 = store.save_edited_version("CASE-001", "SESSION-001", "Bài đã sửa lần 1", 12.5, 45.0)
        assert v2 == 2

        sample_eval = Evaluation(
            schema_version="1.0",
            case_id="CASE-001",
            session_id="SESSION-001",
            evaluation_id="EVAL-001",
            article_version=1,
            rubric_version="v1.0",
            model="gemma-4-12b",
            claims=[
                Claim(
                    claim_id="CLM-001",
                    text="Lô A thu hoạch ngày 20/10.",
                    span=(0, 26),
                    evidence_status=EvidenceStatus.SUPPORTED,
                    reason="Khớp nhật ký",
                    failure_codes=[],
                    evidence_anchors=[
                        Anchor(
                            source_id="SRC-001",
                            record_id="REC-001",
                            excerpt="Ngày 20/10: thu hoạch lô A.",
                            valid=True,
                        )
                    ],
                )
            ],
            completeness=[],
            tone_warnings=[],
            editing=EditingMetrics(edit_ratio=12.5, active_review_seconds=45.0),
            latency_s=1.25,
            run_status=RunStatus.COMPLETED,
        )

        store.save_evaluation(sample_eval)
        loaded = store.load_evaluation("EVAL-001")
        assert loaded is not None
        assert loaded.claims[0].claim_id == "CLM-001"
        assert loaded.claims[0].evidence_anchors[0].record_id == "REC-001"
        assert meta1["article_hash"] == store.compute_sha256("Bài gốc ban đầu")


def test_pipeline_exceptions_table_26():
    """Kiểm thử DoD pipeline: xử lý ngoại lệ theo Bảng 26 (bài rỗng, evidence rỗng, không sửa bài)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = StorageManager(db_path=str(Path(tmpdir) / "test_pipe.db"))
        pipe = EvaluationPipeline(storage=store)

        # 1. Bài viết rỗng -> chặn không chạy pipeline
        with pytest.raises(PipelineInputError):
            pipe.run("", [EvidenceRecord(source_id="S1", record_id="R1", text="Nhật ký")])

        # 2. Evidence rỗng nhưng chưa xác nhận -> chặn theo UC-01 EX1
        with pytest.raises(PipelineInputError):
            pipe.run("Bài viết có nội dung.", [], allow_empty_evidence=False)

        # 3. Evidence rỗng nhưng người dùng xác nhận -> chạy và đánh dấu INSUFFICIENT_EVIDENCE
        eval_res = pipe.run("Lô A thu hoạch ngày 20/10.", [], allow_empty_evidence=True)
        assert eval_res.run_status == RunStatus.COMPLETED
        assert len(eval_res.claims) >= 1
        assert eval_res.claims[0].evidence_status == EvidenceStatus.INSUFFICIENT_EVIDENCE
        assert FailureCode.ERR_UNSUPPORTED in eval_res.claims[0].failure_codes

        # 4. Người dùng bấm lưu nhưng không sửa nội dung -> Edit Ratio = 0%, không tạo version giả
        edit_rep = pipe.record_user_edit(
            eval_res.case_id,
            eval_res.session_id,
            "Lô A thu hoạch ngày 20/10.",
            "Lô A thu hoạch ngày 20/10.",
            20.0,
        )
        assert edit_rep["unchanged"] is True
        assert edit_rep["new_version"] is None
        assert edit_rep["metrics"].edit_ratio == 0.0