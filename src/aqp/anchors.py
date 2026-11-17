from .schemas import Anchor, EvidenceRecord

def validate_anchors(anchors: list[Anchor], evidence_records: list[EvidenceRecord]) -> list[Anchor]:
    """Kiểm tra excerpt có thật trong record không (substring/rapidfuzz)"""
    pass
