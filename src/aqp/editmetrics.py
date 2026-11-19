import re
from typing import Dict, Any, List
from rapidfuzz.distance import Levenshtein
from .schemas import EditingMetrics


def tokenize_words(text: str) -> List[str]:
    """Tách văn bản thành danh sách các từ (token) phục vụ tính khoảng cách cấp từ."""
    if not text or not text.strip():
        return []
    return re.findall(r"\S+", text.strip())


def compute_word_diff_counts(initial_tokens: List[str], edited_tokens: List[str]) -> Dict[str, int]:
    """
    Đếm số từ thêm (insertions), xóa (deletions), sửa (substitutions)
    và khoảng cách chỉnh sửa cấp từ (word edit distance) phục vụ hiển thị UC-03.
    """
    opcodes = Levenshtein.opcodes(initial_tokens, edited_tokens)
    insertions = 0
    deletions = 0
    substitutions = 0

    for op in opcodes:
        if op.tag == "insert":
            insertions += (op.dest_end - op.dest_start)
        elif op.tag == "delete":
            deletions += (op.src_end - op.src_start)
        elif op.tag == "replace":
            src_len = op.src_end - op.src_start
            dest_len = op.dest_end - op.dest_start
            sub = min(src_len, dest_len)
            substitutions += sub
            if dest_len > src_len:
                insertions += (dest_len - src_len)
            elif src_len > dest_len:
                deletions += (src_len - dest_len)

    word_distance = Levenshtein.distance(initial_tokens, edited_tokens)
    return {
        "word_distance": word_distance,
        "inserted_words": insertions,
        "deleted_words": deletions,
        "substituted_words": substitutions,
        "initial_word_count": len(initial_tokens),
        "edited_word_count": len(edited_tokens),
    }


def calculate_edit_ratio(
    initial_text: str,
    edited_text: str,
    active_review_seconds: float = 0.0,
) -> EditingMetrics:
    """
    Tính chỉ số Edit Ratio (%) theo công thức (4) mục 5.7.1:
    EditRatio(T_initial, T_edited) = WordEditDistance(T_initial, T_edited) / max(1, |T_edited|) * 100%
    """
    if initial_text.strip() == edited_text.strip():
        return EditingMetrics(
            edit_ratio=0.0,
            active_review_seconds=round(max(0.0, active_review_seconds), 2),
        )

    initial_tokens = tokenize_words(initial_text)
    edited_tokens = tokenize_words(edited_text)

    word_distance = Levenshtein.distance(initial_tokens, edited_tokens)
    denominator = max(1, len(edited_tokens))
    ratio = round((word_distance / denominator) * 100.0, 2)

    return EditingMetrics(
        edit_ratio=ratio,
        active_review_seconds=round(max(0.0, active_review_seconds), 2),
    )


def get_detailed_edit_report(
    initial_text: str,
    edited_text: str,
    active_review_seconds: float = 0.0,
) -> Dict[str, Any]:
    """
    Trả về cả đối tượng EditingMetrics lẫn thống kê số từ thêm/xóa/sửa và cờ unchanged
    để phục vụ UC-03 (bao gồm luồng ngoại lệ EX1 khi văn bản không thay đổi).
    """
    unchanged = (initial_text.strip() == edited_text.strip())
    metrics = calculate_edit_ratio(initial_text, edited_text, active_review_seconds)
    diff_counts = compute_word_diff_counts(
        tokenize_words(initial_text),
        tokenize_words(edited_text),
    )
    return {
        "metrics": metrics,
        "unchanged": unchanged,
        **diff_counts,
    }