from .schemas import Claim

def extract_claims(article: str) -> list[Claim]:
    """LLM tách atomic claim, giữ nguyên câu gốc và vị trí span"""
    pass
