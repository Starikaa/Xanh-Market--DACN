from .schemas import Claim, EvidenceRecord, EvidenceStatus, Anchor

def verify_claim(claim: Claim, evidence: list[EvidenceRecord]) -> tuple[EvidenceStatus, str, list[Anchor]]:
    """Rule tất định (số/ngày/đơn vị) + LLM judge để gán nhãn D1"""
    pass
