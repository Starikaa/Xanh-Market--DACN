from .schemas import EditingMetrics

def calculate_edit_ratio(initial_text: str, edited_text: str, active_review_seconds: float) -> EditingMetrics:
    """Tính Edit Ratio bằng Levenshtein cấp token và thời gian rà soát"""
    pass
