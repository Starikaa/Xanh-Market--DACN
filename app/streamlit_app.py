import json
import sys
import time
import uuid
from pathlib import Path
from typing import List

import streamlit as st

# Đảm bảo import được gói aqp từ thư mục src/ khi chạy lệnh `streamlit run app/streamlit_app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aqp.schemas import (
    Evaluation,
    EvidenceRecord,
    EvidenceStatus,
    RunStatus,
)
from aqp.pipeline import EvaluationPipeline, PipelineInputError, LLMServiceError
from aqp.storage import StorageManager


SAMPLE_BENCHMARK_CASES = {
    "CASE-001: Cam sành Vĩnh Long (Có lỗi sai số liệu & thiếu nguồn)": {
        "product_type": "specialty_fruit",
        "article": (
            "Lô cam sành A được thu hoạch ngày 20/10 tại vườn canh tác chuẩn. "
            "Trái cam mọng nước với độ ngọt Brix đạt 16% vượt trội. "
            "Sản phẩm đã đạt chứng nhận Organic quốc tế giúp chữa dứt bệnh cảm cúm."
        ),
        "evidence": [
            {
                "source_id": "SRC-DIARY-01",
                "record_id": "REC-001",
                "text": "Ngày 20/10: thu hoạch lô cam sành A tại vườn canh tác chuẩn, sản lượng 500kg.",
                "meta": {"date": "2026-10-20", "type": "diary"},
            },
            {
                "source_id": "SRC-IOT-02",
                "record_id": "REC-002",
                "text": "Kết quả đo mẫu lô cam sành A: độ ngọt Brix đạt 12%, độ ẩm đất 75%.",
                "meta": {"date": "2026-10-20", "type": "sensor"},
            },
        ],
    },
    "CASE-002: Rau cải ngọt VietGAP (Bám sát bằng chứng)": {
        "product_type": "fresh_produce",
        "article": (
            "Lô rau cải ngọt B được thu hoạch ngày 12/11 với quy trình canh tác an toàn. "
            "Sản phẩm đạt chứng nhận VietGAP theo hồ sơ nông trại."
        ),
        "evidence": [
            {
                "source_id": "SRC-DIARY-02",
                "record_id": "REC-010",
                "text": "Ngày 12/11: thu hoạch lô rau cải ngọt B với quy trình canh tác an toàn.",
                "meta": {"date": "2026-11-12", "type": "diary"},
            },
            {
                "source_id": "SRC-CERT-01",
                "record_id": "REC-011",
                "text": "Giấy chứng nhận VietGAP cấp cho sản phẩm lô rau cải ngọt B theo hồ sơ nông trại.",
                "meta": {"date": "2026-06-01", "type": "certificate"},
            },
        ],
    },
}


def init_session_state() -> None:
    defaults = {
        "session_id": f"SESSION-{uuid.uuid4().hex[:6].upper()}",
        "case_id": f"CASE-{uuid.uuid4().hex[:6].upper()}",
        "product_type": "fresh_produce",
        "article_input": "",
        "evidence_input": "",
        "parsed_evidence": [],
        "input_confirmed": False,
        "allow_empty_evidence": False,
        "article_version": 1,
        "current_evaluation": None,
        "previous_evaluation": None,
        "selected_claim_id": None,
        "working_article_text": "",
        "review_start_ts": None,
        "accumulated_review_sec": 0.0,
        "last_edit_report": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def parse_evidence_text(raw_text: str) -> List[EvidenceRecord]:
    """Hỗ trợ nạp evidence dạng mảng JSON hoặc từng dòng văn bản."""
    cleaned = raw_text.strip()
    if not cleaned:
        return []
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        return [EvidenceRecord(**item) for item in data]
    except Exception:
        records: List[EvidenceRecord] = []
        for idx, line in enumerate(cleaned.splitlines(), start=1):
            if line.strip():
                records.append(
                    EvidenceRecord(
                        source_id="SRC-MANUAL",
                        record_id=f"REC-{idx:03d}",
                        text=line.strip(),
                        meta={"type": "manual_line"},
                    )
                )
        return records


def render_highlighted_article(eval_data: Evaluation) -> str:
    """Tô màu từng claim theo nhãn D1 để hiển thị trực quan ở Bước 2 (UC-02)."""
    color_map = {
        EvidenceStatus.SUPPORTED: ("#d4edda", "#155724"),
        EvidenceStatus.CONTRADICTORY: ("#f8d7da", "#721c24"),
        EvidenceStatus.INSUFFICIENT_EVIDENCE: ("#fff3cd", "#856404"),
        EvidenceStatus.NEEDS_REVIEW: ("#e2e3e5", "#383d41"),
    }
    html_parts = []
    for clm in eval_data.claims:
        bg, fg = color_map.get(clm.evidence_status, ("#f8f9fa", "#212529"))
        codes_badge = (
            f" <code>[{', '.join(c.value for c in clm.failure_codes)}]</code>"
            if clm.failure_codes
            else ""
        )
        html_parts.append(
            f"<span style='background-color:{bg}; color:{fg}; padding:4px 8px; "
            f"border-radius:4px; display:inline-block; margin:3px 0;'>"
            f"<b>[{clm.claim_id}]</b> {clm.text} "
            f"<i>({clm.evidence_status.value})</i>{codes_badge}</span>"
        )
    return "<div style='line-height:1.9;'>" + "<br/>".join(html_parts) + "</div>"


def main() -> None:
    st.set_page_config(
        page_title="Xanh Story CMS - Kiểm định Chất lượng Nội dung Nông sản AI (PoC)",
        layout="wide",
    )
    init_session_state()

    storage = StorageManager()
    pipeline = EvaluationPipeline(storage=storage)

    st.title("Hệ thống Đánh giá và Hỗ trợ Cải thiện Nội dung Nông sản do LLM tạo (PoC)")
    st.caption(
        f"Mã phiên: `{st.session_state.session_id}` | "
        f"Mã ca: `{st.session_state.case_id}` | "
        f"Phiên bản bài viết: `v{st.session_state.article_version}`"
    )

    tab1, tab2, tab3 = st.tabs([
        "Bước 1: Nạp Nội dung & Bằng chứng (UC-01)",
        "Bước 2: Đánh giá & Chẩn đoán lỗi (UC-02)",
        "Bước 3: Hiệu đính & Xuất kết quả (UC-03 / UC-04)",
    ])

    # =========================================================================
    # BƯỚC 1: NẠP NỘI DUNG VÀ DỮ LIỆU BẰNG CHỨNG (UC-01)
    # =========================================================================
    with tab1:
        st.subheader("UC-01: Nạp bài viết truyền thông AI và dữ liệu bằng chứng gốc")

        col_sample, col_type = st.columns([2, 1])
        with col_sample:
            selected_sample = st.selectbox(
                "Chọn dữ liệu mẫu từ tập Benchmark (UC-01 AC1):",
                options=["-- Tự nhập dữ liệu mới --"] + list(SAMPLE_BENCHMARK_CASES.keys()),
            )
            if st.button("Nạp ca mẫu từ Benchmark"):
                if selected_sample != "-- Tự nhập dữ liệu mới --":
                    sample_data = SAMPLE_BENCHMARK_CASES[selected_sample]
                    st.session_state.case_id = selected_sample.split(":")[0].strip()
                    st.session_state.product_type = sample_data["product_type"]
                    st.session_state.article_input = sample_data["article"]
                    st.session_state.evidence_input = json.dumps(
                        sample_data["evidence"], ensure_ascii=False, indent=2
                    )
                    st.session_state.article_version = 1
                    st.session_state.current_evaluation = None
                    st.session_state.previous_evaluation = None
                    st.rerun()

        with col_type:
            st.session_state.product_type = st.selectbox(
                "Nhóm sản phẩm nông sản (cấu hình Checklist D2):",
                options=["fresh_produce", "specialty_fruit", "processed"],
                index=["fresh_produce", "specialty_fruit", "processed"].index(
                    st.session_state.product_type
                ),
            )

        col_left, col_right = st.columns(2)
        with col_left:
            st.session_state.article_input = st.text_area(
                "1. Nội dung bài viết truyền thông do AI tạo (AI Draft):",
                value=st.session_state.article_input,
                height=240,
                placeholder="Dán bài viết giới thiệu nông sản do LLM tạo vào đây...",
            )
        with col_right:
            st.session_state.evidence_input = st.text_area(
                "2. Dữ liệu bằng chứng gốc (JSON có record_id hoặc từng dòng nhật ký):",
                value=st.session_state.evidence_input,
                height=240,
                placeholder='[{"source_id": "SRC-001", "record_id": "REC-001", "text": "..."}]',
            )

        st.session_state.allow_empty_evidence = st.checkbox(
            "Xác nhận tiếp tục ngay cả khi không có dữ liệu bằng chứng (Bảng 26: đánh dấu mọi luận điểm là chưa kiểm chứng)",
            value=st.session_state.allow_empty_evidence,
        )

        if st.button("Xác nhận nạp dữ liệu", type="primary"):
            parsed_ev = parse_evidence_text(st.session_state.evidence_input)
            if not parsed_ev and st.session_state.allow_empty_evidence:
                st.warning(
                    "Cảnh báo: Bạn đang nạp bài viết mà không có bằng chứng nguồn. "
                    "Hệ thống sẽ không kết luận các luận điểm sai thực tế mà đánh dấu là INSUFFICIENT_EVIDENCE."
                )
            try:
                meta = pipeline.validate_and_prepare_input(
                    article=st.session_state.article_input,
                    evidence=parsed_ev,
                    allow_empty_evidence=st.session_state.allow_empty_evidence,
                    case_id=st.session_state.case_id,
                    session_id=st.session_state.session_id,
                    product_type=st.session_state.product_type,
                )
                st.session_state.parsed_evidence = parsed_ev
                st.session_state.input_confirmed = True
                st.session_state.working_article_text = st.session_state.article_input
                st.success(
                    f"Dữ liệu đã được chuẩn hóa và lưu bất biến! "
                    f"(Hash bài gốc: `{meta['article_hash'][:12]}...`). "
                    f"Hãy chuyển sang Bước 2 để chạy đánh giá."
                )
            except PipelineInputError as err:
                st.session_state.input_confirmed = False
                st.error(str(err))

    # =========================================================================
    # BƯỚC 2: ĐÁNH GIÁ VÀ CHẨN ĐOÁN LỖI NỘI DUNG (UC-02)
    # =========================================================================
    with tab2:
        st.subheader("UC-02: Đánh giá đa chiều và Chẩn đoán lỗi ở cấp luận điểm (Claim-level)")

        if not st.session_state.input_confirmed:
            st.info("Vui lòng nạp và xác nhận dữ liệu ở Bước 1 trước khi bắt đầu đánh giá.")
        else:
            if st.button("Bắt đầu đánh giá (Evaluate Content)", type="primary"):
                with st.spinner("Đang tách luận điểm, đối chiếu bằng chứng và chẩn đoán mã lỗi..."):
                    try:
                        eval_res = pipeline.run(
                            article=st.session_state.working_article_text,
                            evidence=st.session_state.parsed_evidence,
                            case_id=st.session_state.case_id,
                            session_id=st.session_state.session_id,
                            article_version=st.session_state.article_version,
                            product_type=st.session_state.product_type,
                            allow_empty_evidence=st.session_state.allow_empty_evidence,
                        )
                        st.session_state.current_evaluation = eval_res
                        st.session_state.review_start_ts = time.perf_counter()
                    except LLMServiceError as api_err:
                        st.error(str(api_err))

            eval_data: Evaluation = st.session_state.current_evaluation
            if eval_data:
                if eval_data.run_status == RunStatus.EVALUATION_FAILED:
                    st.error(
                        "EVALUATION_FAILED: Đầu ra từ mô hình không khớp schema quy định. "
                        "Đã ghi log kỹ thuật và ẩn dữ liệu lỗi theo Bảng 26."
                    )
                else:
                    # Quy tắc an toàn mục 5.2.2: Hiển thị trạng thái "CẦN RÀ SOÁT" độc lập
                    has_contradiction = any(
                        c.evidence_status == EvidenceStatus.CONTRADICTORY for c in eval_data.claims
                    )
                    has_unsupported = any(
                        c.evidence_status in (EvidenceStatus.INSUFFICIENT_EVIDENCE, EvidenceStatus.NEEDS_REVIEW)
                        for c in eval_data.claims
                    )
                    if has_contradiction or has_unsupported or eval_data.tone_warnings:
                        st.warning(
                            "TRẠNG THÁI AN TOÀN: CẦN BIÊN TẬP VIÊN RÀ SOÁT "
                            "(Phát hiện luận điểm mâu thuẫn, chưa đủ bằng chứng hoặc cảnh báo văn phong)."
                        )
                    else:
                        st.success("Tất cả luận điểm đều có bằng chứng hợp lệ hỗ trợ.")

                    # Thống kê nhanh D1 - D4
                    total_c = max(1, len(eval_data.claims))
                    sup_c = sum(1 for c in eval_data.claims if c.evidence_status == EvidenceStatus.SUPPORTED)
                    con_c = sum(1 for c in eval_data.claims if c.evidence_status == EvidenceStatus.CONTRADICTORY)
                    ins_c = sum(1 for c in eval_data.claims if c.evidence_status == EvidenceStatus.INSUFFICIENT_EVIDENCE)
                    valid_anchor_c = sum(
                        1 for c in eval_data.claims
                        if any(a.valid is not False for a in c.evidence_anchors)
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("D1 - Tỷ lệ có bằng chứng", f"{(sup_c / total_c) * 100:.1f}% ({sup_c}/{len(eval_data.claims)})")
                    m2.metric("D1 - Mâu thuẫn / Thiếu nguồn", f"{con_c} mâu thuẫn | {ins_c} thiếu")
                    m3.metric("D4 - Truy vết Anchor hợp lệ", f"{(valid_anchor_c / total_c) * 100:.1f}%")
                    m4.metric("Thời gian xử lý (Latency)", f"{eval_data.latency_s:.2f} giây")

                    st.markdown("#### Bài viết được đánh dấu theo trạng thái bằng chứng (Highlight)")
                    st.markdown(render_highlighted_article(eval_data), unsafe_allow_html=True)

                    st.markdown("#### Chi tiết từng luận điểm, Truy vết Anchor và Gợi ý sửa đổi (UC-02 AC1)")
                    for idx, clm in enumerate(eval_data.claims):
                        with st.expander(
                            f"{clm.claim_id}: {clm.text} — [{clm.evidence_status.value}]",
                            expanded=(clm.evidence_status != EvidenceStatus.SUPPORTED),
                        ):
                            st.write(f"**Lý do chẩn đoán:** {clm.reason}")
                            st.write(
                                f"**Mã lỗi biên tập (Failure Taxonomy):** "
                                f"{', '.join(c.value for c in clm.failure_codes) if clm.failure_codes else 'Không có lỗi'}"
                            )

                            if clm.evidence_anchors:
                                st.markdown("**Đoạn trích bằng chứng gốc (Traceability Anchor):**")
                                for anc in clm.evidence_anchors:
                                    st.code(
                                        f"[{anc.source_id} | {anc.record_id}] \"{anc.excerpt}\" (Hợp lệ: {anc.valid})",
                                        language="text",
                                    )
                            else:
                                st.caption("Không có anchor bằng chứng tương ứng.")

                            if clm.suggestion:
                                st.info(
                                    f"**Gợi ý hành động (`{clm.suggestion.action_type}`):** "
                                    f"{clm.suggestion.text or 'Cần bổ sung nguồn hoặc lược bỏ luận điểm.'}\n\n"
                                    f"*Lưu ý giới hạn:* {clm.suggestion.limit_note}"
                                )
                                c_acc, c_edit, c_ign = st.columns(3)
                                if c_acc.button("Chấp nhận (Accept)", key=f"acc_{clm.claim_id}_{idx}"):
                                    clm.suggestion.user_decision = "accepted"
                                    if clm.suggestion.text:
                                        st.session_state.working_article_text = (
                                            st.session_state.working_article_text.replace(
                                                clm.text, clm.suggestion.text
                                            )
                                        )
                                    st.success("Đã áp dụng gợi ý vào khung biên tập ở Bước 3.")
                                if c_edit.button("Tùy chỉnh (Edit)", key=f"edt_{clm.claim_id}_{idx}"):
                                    clm.suggestion.user_decision = "edited"
                                    st.info("Hãy chỉnh sửa trực tiếp câu văn tại khung biên tập ở Bước 3.")
                                if c_ign.button("Bỏ qua (Ignore)", key=f"ign_{clm.claim_id}_{idx}"):
                                    clm.suggestion.user_decision = "ignored"
                                    st.caption("Đã đánh dấu bỏ qua gợi ý này.")

                    col_d2, col_d3 = st.columns(2)
                    with col_d2:
                        st.markdown("#### D2 – Kiểm tra tính đầy đủ (Checklist)")
                        if eval_data.completeness:
                            for item in eval_data.completeness:
                                st.write(f"- **{item.field}** (`{item.status}`): {item.description}")
                        else:
                            st.caption("Chưa ghi nhận cảnh báo thiếu trường từ module completeness.py.")

                    with col_d3:
                        st.markdown("#### D3 – Văn phong & Cảnh báo nội dung")
                        if eval_data.tone_warnings:
                            for warn in eval_data.tone_warnings:
                                st.warning(f"**{warn.phrase}** ({warn.severity}): {warn.reason}")
                        else:
                            st.caption("Chưa ghi nhận cảnh báo từ ngữ phóng đại từ module tone.py.")

    # =========================================================================
    # BƯỚC 3: HIỆU ĐÍNH BÀI VIẾT, TÁI ĐÁNH GIÁ & XUẤT LOG (UC-03 & UC-04)
    # =========================================================================
    with tab3:
        st.subheader("UC-03 & UC-04: Hiệu đính bài viết, Đo phản hồi ngầm và Xuất kết quả thực nghiệm")

        if not st.session_state.current_evaluation:
            st.info("Vui lòng chạy đánh giá ở Bước 2 trước khi hiệu đính và xuất dữ liệu.")
        else:
            st.markdown("##### Khung biên tập văn bản (Chỉnh sửa trực tiếp câu chữ)")
            edited_text = st.text_area(
                "Nội dung bản thảo hiệu đính:",
                value=st.session_state.working_article_text,
                height=200,
            )
            st.session_state.working_article_text = edited_text

            if st.button("Lưu bản hiệu đính (Save & Compute Metrics)", type="primary"):
                now_ts = time.perf_counter()
                elapsed = 0.0
                if st.session_state.review_start_ts is not None:
                    elapsed = now_ts - st.session_state.review_start_ts
                total_review_sec = round(st.session_state.accumulated_review_sec + elapsed, 2)
                st.session_state.accumulated_review_sec = total_review_sec
                st.session_state.review_start_ts = now_ts

                edit_res = pipeline.record_user_edit(
                    case_id=st.session_state.case_id,
                    session_id=st.session_state.session_id,
                    initial_article=st.session_state.article_input,
                    edited_article=edited_text,
                    active_review_seconds=total_review_sec,
                )
                st.session_state.last_edit_report = edit_res
                st.session_state.current_evaluation.editing = edit_res["metrics"]
                storage.save_evaluation(st.session_state.current_evaluation)

                if edit_res["unchanged"]:
                    st.warning(edit_res["message"])
                else:
                    st.session_state.article_version = edit_res["new_version"]
                    st.success(edit_res["message"])

            if st.session_state.last_edit_report:
                rep = st.session_state.last_edit_report
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Tỷ lệ chỉnh sửa (Edit Ratio)", f"{rep['metrics'].edit_ratio:.2f}%")
                e2.metric("Thời gian rà soát (Dwell Proxy)", f"{rep['metrics'].active_review_seconds:.2f} giây")
                e3.metric("Số từ Thêm / Xóa", f"+{rep['inserted_words']} / -{rep['deleted_words']}")
                e4.metric("Số từ Thay thế (Sửa)", f"{rep['substituted_words']} từ")

            st.divider()
            st.markdown("##### Tái đánh giá sau hiệu đính (FR11 / UC-03 AC1)")
            if st.button("Đánh giá lại văn bản mới (Re-evaluate)"):
                st.session_state.previous_evaluation = st.session_state.current_evaluation
                with st.spinner("Đang chạy lại pipeline trên phiên bản văn bản mới..."):
                    new_eval = pipeline.run(
                        article=st.session_state.working_article_text,
                        evidence=st.session_state.parsed_evidence,
                        case_id=st.session_state.case_id,
                        session_id=st.session_state.session_id,
                        article_version=st.session_state.article_version,
                        product_type=st.session_state.product_type,
                        allow_empty_evidence=st.session_state.allow_empty_evidence,
                        editing_metrics=(
                            st.session_state.last_edit_report["metrics"]
                            if st.session_state.last_edit_report
                            else None
                        ),
                        is_reevaluation=True,
                    )
                    st.session_state.current_evaluation = new_eval

                prev = st.session_state.previous_evaluation
                curr = st.session_state.current_evaluation
                if prev and curr:
                    prev_err = sum(len(c.failure_codes) for c in prev.claims)
                    curr_err = sum(len(c.failure_codes) for c in curr.claims)
                    st.success(
                        f"Hoàn tất tái đánh giá (`{curr.evaluation_id}`)! "
                        f"Tổng số mã lỗi giảm từ **{prev_err}** xuống **{curr_err}**."
                    )

            st.divider()
            st.markdown("##### Xuất kết quả kiểm định thực nghiệm (UC-04)")
            curr_eval: Evaluation = st.session_state.current_evaluation
            json_data = storage.export_json(curr_eval)
            csv_data = storage.export_csv(curr_eval)

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="Tải xuống Log JSON đầy đủ (UC-04 Chính)",
                    data=json_data.encode("utf-8"),
                    file_name=f"{curr_eval.case_id}_{curr_eval.evaluation_id}.json",
                    mime="application/json",
                )
            with dl_col2:
                st.download_button(
                    label="Tải xuống Bảng tóm tắt CSV (UC-04 AC1)",
                    data=csv_data.encode("utf-8-sig"),
                    file_name=f"{curr_eval.case_id}_{curr_eval.evaluation_id}_summary.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()