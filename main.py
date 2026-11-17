from aqp.schemas import EvidenceRecord, EvidenceStatus, RunStatus
from aqp.pipeline import EvaluationPipeline

if __name__ == "__main__":
    # Khởi tạo dữ liệu mẫu kiểm tra schema
    sample_evidence = [
        EvidenceRecord(
            source_id="SRC-001",
            record_id="REC-001",
            text="Thu hoạch ngày 20/10, độ ngọt Brix 14%."
        )
    ]
    sample_article = "Sản phẩm thu hoạch ngày 20/10 với vị ngọt tự nhiên."

    pipeline = EvaluationPipeline()
    result = pipeline.run(sample_article, sample_evidence)

    print("Khởi tạo thành công!")
    print("Dữ liệu bằng chứng mẫu:", sample_evidence[0].model_dump_json(indent=2))
    print("Kết quả pipeline (chưa hiện thực logic):", result)