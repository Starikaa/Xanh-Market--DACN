from .schemas import Claim, EvidenceRecord

def retrieve_evidence(claim: Claim, evidence_records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    """Đưa evidence vào prompt (hoặc dùng BM25 mở rộng sau)"""
    pass
