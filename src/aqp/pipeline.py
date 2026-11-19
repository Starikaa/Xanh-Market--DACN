import re
import time
import uuid
from enum import Enum
from typing import List, Optional, Dict, Any, Callable

from pydantic import ValidationError

from .schemas import (
    Anchor,
    ChecklistItem,
    Claim,
    EditingMetrics,
    Evaluation,
    EvidenceRecord,
    EvidenceStatus,
    FailureCode,
    RunStatus,
    Suggestion,
    ToneWarning,
)
from . import (
    normalize,
    claims as claims_mod,
    retrieve as retrieve_mod,
    verify as verify_mod,
    anchors as anchors_mod,
    diagnose as diagnose_mod,
    completeness as completeness_mod,
    tone as tone_mod,
    recommend as recommend_mod,
)
from .editmetrics import get_detailed_edit_report
from .storage import StorageManager


class SessionState(str, Enum):
    """Trạng thái phiên theo mục 5.5.4 của báo cáo."""
    DRAFT = "DRAFT"
    READY = "READY"
    EVALUATING = "EVALUATING"
    EVALUATED = "EVALUATED"
    EDITED = "EDITED"
    RE_EVALUATING = "RE_EVALUATING"


class PipelineInputError(ValueError):
    """Ngoại lệ khi dữ liệu đầu vào rỗng hoặc sai định dạng (EX1 của UC-01, Bảng 26)."""
    pass


class LLMServiceError(RuntimeError):
    """Ngoại lệ khi dịch vụ LLM timeout hoặc hết quota sau khi đã retry 3 lần (EX1 của UC-02)."""
    pass


class SchemaOutputError(RuntimeError):
    """Ngoại lệ khi đầu ra từ mô hình sai schema (Bảng 26 -> EVALUATION_FAILED)."""
    pass


class EvaluationPipeline:
    """
    Điều phối phiên làm việc và ghép nối toàn bộ các bước trong luồng xử lý (Bảng 25 & Bảng 26).
    """

    def __init__(
        self,
        storage: Optional[StorageManager] = None,
        model_name: str = "gemma-4-12b-it-Q4_K_M.gguf",
        rubric_version: str = "v1.0",
        max_retries: int = 3,
    ):
        self.storage = storage or StorageManager()
        self.model_name = model_name
        self.rubric_version = rubric_version
        self.max_retries = max_retries
        self.state: SessionState = SessionState.DRAFT

    def validate_and_prepare_input(
        self,
        article: str,
        evidence: List[EvidenceRecord],
        allow_empty_evidence: bool = False,
        case_id: Optional[str] = None,
        session_id: Optional[str] = None,
        product_type: str = "fresh_produce",
    ) -> Dict[str, Any]:
        """
        Bước 1 & 2 (Bảng 25) + Xử lý ngoại lệ Bảng 26:
        - Bài viết rỗng: chặn không chạy pipeline, báo trường cần sửa.
        - Evidence rỗng: yêu cầu người dùng xác nhận; nếu chưa xác nhận thì báo lỗi theo UC-01 EX1.
        """
        if not isinstance(article, str) or not article.strip():
            raise PipelineInputError(
                "Vui lòng cung cấp đầy đủ cả bài viết AI và dữ liệu bằng chứng trước khi tiếp tục."
            )

        if not evidence and not allow_empty_evidence:
            raise PipelineInputError(
                "Vui lòng cung cấp đầy đủ cả bài viết AI và dữ liệu bằng chứng trước khi tiếp tục."
            )

        # Gọi normalize của Bạn 1 nếu đã hiện thực, nếu còn pass (None) thì dùng chuẩn hóa mặc định
        norm_res = normalize.normalize_input(article, evidence)
        if not isinstance(norm_res, dict):
            resolved_case_id = case_id or f"CASE-{uuid.uuid4().hex[:6].upper()}"
            resolved_session_id = session_id or f"SESSION-{uuid.uuid4().hex[:6].upper()}"
        else:
            resolved_case_id = norm_res.get("case_id") or case_id or f"CASE-{uuid.uuid4().hex[:6].upper()}"
            resolved_session_id = norm_res.get("session_id") or session_id or f"SESSION-{uuid.uuid4().hex[:6].upper()}"

        meta = self.storage.save_initial_case(
            case_id=resolved_case_id,
            session_id=resolved_session_id,
            initial_article=article,
            evidence_records=evidence,
            product_type=product_type,
        )
        self.state = SessionState.READY
        return meta

    def _call_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Thử lại (retry) tối đa 3 lần khi gặp lỗi kết nối/timeout tới LLM API (UC-02 EX1).
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (TimeoutError, ConnectionError) as exc:
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(0.1 * attempt)
        raise LLMServiceError(
            "Không thể kết nối với dịch vụ LLM-as-a-Judge. Vui lòng kiểm tra lại API key hoặc kết nối mạng."
        ) from last_err

    @staticmethod
    def _fallback_extract_claims(article: str) -> List[Claim]:
        """
        Fallback tách câu khi module claims.py của Bạn 1 chưa hiện thực (đang trả về None),
        giúp Bạn 2 phát triển và test giao diện Streamlit + luồng dữ liệu độc lập.
        """
        claims_list: List[Claim] = []
        pattern = re.compile(r"[^.!?\n]+[.!?]?")
        idx = 1
        for match in pattern.finditer(article):
            seg = match.group(0).strip()
            if not seg:
                continue
            claims_list.append(
                Claim(
                    claim_id=f"CLM-{idx:03d}",
                    text=seg,
                    span=(match.start(), match.end()),
                    evidence_status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
                    reason="Chưa xác minh được từ nguồn hiện có.",
                    failure_codes=[],
                    evidence_anchors=[],
                    suggestion=None,
                )
            )
            idx += 1
        return claims_list

    @staticmethod
    def _fallback_process_claims(
        extracted_claims: List[Claim],
        evidence: List[EvidenceRecord],
    ) -> List[Claim]:
        """
        Fallback đối chiếu, kiểm tra anchor, chẩn đoán mã lỗi và tạo gợi ý theo đúng Bảng 23 & Bảng 27
        khi các hàm của Bạn 1 trả về None.
        """
        processed: List[Claim] = []
        for clm in extracted_claims:
            if not evidence:
                # Bảng 26: Evidence rỗng -> đánh dấu claim chưa kiểm chứng, không tạo anchor giả
                clm.evidence_status = EvidenceStatus.INSUFFICIENT_EVIDENCE
                clm.reason = "Tập bằng chứng rỗng: chưa xác minh được luận điểm từ nguồn hiện có."
                clm.evidence_anchors = []
                clm.failure_codes = [FailureCode.ERR_UNSUPPORTED, FailureCode.ERR_TRACEABILITY]
                clm.suggestion = Suggestion(
                    action_type="add_source",
                    text=None,
                    grounded_in=[],
                    limit_note="Thiếu dữ liệu nguồn; yêu cầu bổ sung hồ sơ hoặc lược bỏ luận điểm.",
                    user_decision="pending",
                )
                processed.append(clm)
                continue

            # Nếu claim chưa được gán anchor/status từ module của Bạn 1, thực hiện đối chiếu từ khóa/số cơ bản
            matched_record: Optional[EvidenceRecord] = None
            clm_numbers = re.findall(r"\d+(?:[.,/]\d+)*", clm.text)

            for rec in evidence:
                # Kiểm tra xem có từ khóa chung hoặc số liệu liên quan không
                common_words = set(re.findall(r"\w+", clm.text.lower())) & set(
                    re.findall(r"\w+", rec.text.lower())
                )
                if len(common_words) >= 3:
                    matched_record = rec
                    break

            if matched_record:
                rec_numbers = re.findall(r"\d+(?:[.,/]\d+)*", matched_record.text)
                if clm_numbers and rec_numbers and not set(clm_numbers).issubset(set(rec_numbers)):
                    clm.evidence_status = EvidenceStatus.CONTRADICTORY
                    clm.reason = (
                        f"Số liệu trong luận điểm ({', '.join(clm_numbers)}) mâu thuẫn với "
                        f"nhật ký nguồn ({', '.join(rec_numbers)})."
                    )
                    anchor = Anchor(
                        source_id=matched_record.source_id,
                        record_id=matched_record.record_id,
                        excerpt=matched_record.text,
                        valid=True,
                    )
                    clm.evidence_anchors = [anchor]
                    clm.failure_codes = [FailureCode.ERR_MISINTERPRETATION]
                    clm.suggestion = Suggestion(
                        action_type="fix_value",
                        text=matched_record.text,
                        grounded_in=[anchor],
                        limit_note="Đề xuất điều chỉnh số liệu khớp với bản ghi nguồn. Bắt buộc biên tập viên xác nhận.",
                        user_decision="pending",
                    )
                else:
                    clm.evidence_status = EvidenceStatus.SUPPORTED
                    clm.reason = "Luận điểm được hỗ trợ trực tiếp bởi bản ghi bằng chứng."
                    anchor = Anchor(
                        source_id=matched_record.source_id,
                        record_id=matched_record.record_id,
                        excerpt=matched_record.text,
                        valid=True,
                    )
                    clm.evidence_anchors = [anchor]
                    clm.failure_codes = []
                    clm.suggestion = None
            else:
                clm.evidence_status = EvidenceStatus.INSUFFICIENT_EVIDENCE
                clm.reason = "Chưa tìm thấy thông tin đối chứng trong tập dữ liệu nguồn hiện có."
                clm.evidence_anchors = []
                clm.failure_codes = [FailureCode.ERR_UNSUPPORTED, FailureCode.ERR_TRACEABILITY]
                clm.suggestion = Suggestion(
                    action_type="add_source",
                    text=None,
                    grounded_in=[],
                    limit_note="Không tự tạo dữ kiện khi nguồn thiếu. Yêu cầu bổ sung nguồn hoặc bỏ luận điểm.",
                    user_decision="pending",
                )

            processed.append(clm)
        return processed

    def run(
        self,
        article: str,
        evidence: List[EvidenceRecord],
        case_id: Optional[str] = None,
        session_id: Optional[str] = None,
        article_version: int = 1,
        product_type: str = "fresh_produce",
        allow_empty_evidence: bool = False,
        editing_metrics: Optional[EditingMetrics] = None,
        is_reevaluation: bool = False,
    ) -> Evaluation:
        """
        Điều phối toàn bộ các bước:
        Nạp & Chuẩn hóa -> Tách claim -> Đối chiếu -> Kiểm tra Anchor -> Chẩn đoán & Checklist/Tone -> Gợi ý -> Lưu DB.
        """
        start_time = time.perf_counter()

        meta = self.validate_and_prepare_input(
            article=article,
            evidence=evidence,
            allow_empty_evidence=allow_empty_evidence,
            case_id=case_id,
            session_id=session_id,
            product_type=product_type,
        )
        resolved_case_id = meta["case_id"]
        resolved_session_id = meta["session_id"]
        evaluation_id = f"EVAL-{uuid.uuid4().hex[:6].upper()}"

        self.state = SessionState.RE_EVALUATING if is_reevaluation else SessionState.EVALUATING

        try:
            # Bước 3: Tách claim
            extracted_claims = self._call_with_retry(claims_mod.extract_claims, article)
            if extracted_claims is None:
                extracted_claims = self._fallback_extract_claims(article)

            if not isinstance(extracted_claims, list):
                raise SchemaOutputError("Đầu ra bước tách claim không đúng cấu trúc danh sách.")

            # Bước 4 & 5 & 6: Đối chiếu evidence, kiểm tra anchor, chẩn đoán mã lỗi, sinh gợi ý
            if not evidence:
                # Bảng 26: Nếu evidence rỗng và người dùng đã xác nhận tiếp tục
                final_claims = self._fallback_process_claims(extracted_claims, [])
            else:
                ver_implemented = False
                for clm in extracted_claims:
                    rel_ev = retrieve_mod.retrieve_evidence(clm, evidence)
                    if rel_ev is None:
                        rel_ev = evidence

                    ver_res = self._call_with_retry(verify_mod.verify_claim, clm, rel_ev)
                    if ver_res is not None:
                        ver_implemented = True
                        if not isinstance(ver_res, tuple) or len(ver_res) != 3:
                            raise SchemaOutputError("Đầu ra verify_claim sai schema (cần tuple 3 phần tử).")
                        status, reason, raw_anchors = ver_res
                        clm.evidence_status = EvidenceStatus(status)
                        clm.reason = reason
                        validated_anchors = anchors_mod.validate_anchors(raw_anchors, rel_ev)
                        clm.evidence_anchors = validated_anchors if validated_anchors is not None else raw_anchors

                        # Nguyên tắc cứng mục 3 & mục 7: SUPPORTED nhưng không có anchor hợp lệ -> hạ xuống INSUFFICIENT_EVIDENCE + ERR_TRACEABILITY
                        valid_anchors = [a for a in clm.evidence_anchors if a.valid is not False]
                        if clm.evidence_status == EvidenceStatus.SUPPORTED and not valid_anchors:
                            clm.evidence_status = EvidenceStatus.INSUFFICIENT_EVIDENCE
                            if FailureCode.ERR_TRACEABILITY not in clm.failure_codes:
                                clm.failure_codes.append(FailureCode.ERR_TRACEABILITY)

                if ver_implemented:
                    diag_claims = diagnose_mod.diagnose_claims(extracted_claims)
                    if diag_claims is not None:
                        extracted_claims = diag_claims
                    rec_claims = self._call_with_retry(recommend_mod.generate_suggestions, extracted_claims)
                    if rec_claims is not None:
                        extracted_claims = rec_claims
                    final_claims = extracted_claims
                else:
                    final_claims = self._fallback_process_claims(extracted_claims, evidence)

            # Kiểm tra D2 (Completeness) & D3 (Tone)
            comp_items = completeness_mod.check_completeness(product_type, evidence, article)
            if comp_items is None:
                comp_items = []

            tone_items = tone_mod.check_tone(article)
            if tone_items is None:
                tone_items = []

            latency_s = round(time.perf_counter() - start_time, 3)

            eval_result = Evaluation(
                schema_version="1.0",
                case_id=resolved_case_id,
                session_id=resolved_session_id,
                evaluation_id=evaluation_id,
                article_version=article_version,
                rubric_version=self.rubric_version,
                model=self.model_name,
                claims=final_claims,
                completeness=comp_items,
                tone_warnings=tone_items,
                editing=editing_metrics,
                latency_s=latency_s,
                cost_usd=0.0,
                run_status=RunStatus.COMPLETED,
            )

        except (ValidationError, SchemaOutputError, ValueError, TypeError) as exc:
            # Bảng 26: Đầu ra mô hình sai schema -> Đánh dấu EVALUATION_FAILED, lưu log kỹ thuật, không hiển thị dữ liệu lỗi
            latency_s = round(time.perf_counter() - start_time, 3)
            eval_result = Evaluation(
                schema_version="1.0",
                case_id=resolved_case_id,
                session_id=resolved_session_id,
                evaluation_id=evaluation_id,
                article_version=article_version,
                rubric_version=self.rubric_version,
                model=self.model_name,
                claims=[],
                completeness=[],
                tone_warnings=[],
                editing=editing_metrics,
                latency_s=latency_s,
                cost_usd=0.0,
                run_status=RunStatus.EVALUATION_FAILED,
            )
            self.storage.log_trace(
                session_id=resolved_session_id,
                evaluation_id=evaluation_id,
                step_name=f"SCHEMA_ERROR: {type(exc).__name__}",
                model=self.model_name,
                latency_s=latency_s,
            )
            self.storage.save_evaluation(eval_result)
            self.state = SessionState.EVALUATED
            return eval_result

        self.storage.save_evaluation(eval_result)
        self.storage.log_trace(
            session_id=resolved_session_id,
            evaluation_id=evaluation_id,
            step_name="pipeline_run",
            model=self.model_name,
            prompt_version=self.rubric_version,
            latency_s=latency_s,
        )
        self.state = SessionState.EVALUATED
        return eval_result

    def record_user_edit(
        self,
        case_id: str,
        session_id: str,
        initial_article: str,
        edited_article: str,
        active_review_seconds: float,
    ) -> Dict[str, Any]:
        """
        Xử lý Bước 7 (Bảng 25) & UC-03:
        - Nếu người dùng chưa sửa nội dung (Bảng 26 & UC-03 EX1): ghi nhận không có thay đổi,
          Edit Ratio = 0%, không tạo phiên bản sửa giả trong DB.
        - Nếu có sửa: tính Edit Ratio, lưu phiên bản mới vào DB và chuyển trạng thái sang EDITED.
        """
        report = get_detailed_edit_report(initial_article, edited_article, active_review_seconds)
        if report["unchanged"]:
            return {
                "unchanged": True,
                "message": "Không ghi nhận thay đổi nào trên văn bản. Tỷ lệ chỉnh sửa Edit Ratio = 0%",
                "new_version": None,
                **report,
            }

        new_ver = self.storage.save_edited_version(
            case_id=case_id,
            session_id=session_id,
            edited_article=edited_article,
            edit_ratio=report["metrics"].edit_ratio,
            active_review_seconds=report["metrics"].active_review_seconds,
        )
        self.state = SessionState.EDITED
        return {
            "unchanged": False,
            "message": f"Đã lưu bản hiệu đính (Phiên bản v{new_ver}).",
            "new_version": new_ver,
            **report,
        }