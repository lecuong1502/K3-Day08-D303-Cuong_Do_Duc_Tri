"""
RAG Chatbot — University Services.
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm src/ vào sys.path — các module task*.py dùng absolute import
# (vd "from task5_semantic_search import ...") nên cần src/ nằm trong
# sys.path, không phải import kiểu "from src.task9_... import ..." như
# bản gốc (sẽ lỗi vì task9 tự import "from task5_semantic_search import..."
# chứ không phải "from src.task5_semantic_search import...").
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from task9_retrieval_pipeline import retrieve  # noqa: E402
from task10_generation import generate_with_citation  # noqa: E402

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🎓 University Services RAG")
    st.caption("Trợ lý hỏi đáp về dịch vụ và chính sách đại học (học phí, học bổng, ký túc xá, thư viện)")

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Điều kiện xin học bổng Academic Achievement?",
        "Dịch vụ hỗ trợ chỗ ở cho sinh viên như thế nào?",
        "Cách đăng ký học phần qua myRMIT?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)
    use_history = st.checkbox(
        "Nhớ ngữ cảnh hội thoại (follow-up questions)", value=True,
        help="Khi bật, các câu hỏi tiếp theo trong cùng phiên chat sẽ được "
             "LLM diễn giải có ngữ cảnh lượt hỏi trước (KHÔNG ảnh hưởng bước "
             "retrieval, chỉ ảnh hưởng cách LLM trả lời — xem giới hạn trong "
             "docstring generate_with_citation ở src/task10_generation.py)."
    )

    if st.button("🗑️ Xoá lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Kiến trúc hệ thống:**")
    st.caption("Hybrid Retrieval (Semantic + BM25) → RRF Merge → Cross-encoder Rerank → PageIndex Fallback → LLM Generation có Citation (GPT-4o qua OpenRouter)")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def _render_sources(sources: list[dict]):
    """
    Hiển thị source documents đã dùng để trả lời.

    Xử lý linh hoạt vì metadata KHÁC NHAU tuỳ nguồn kết quả:
    - Kết quả 'hybrid' (semantic/BM25, Task 4-7): metadata có 'source', 'type'
    - Kết quả 'pageindex' (fallback, Task 8): metadata có 'source_file',
      'section', 'doc_id' — KHÔNG có key 'source'/'type'
    Nếu chỉ đọc cứng meta.get("source") sẽ hiện "Unknown" sai cho mọi kết quả
    PageIndex fallback dù vẫn có tên file thật trong metadata.
    """
    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            source_name = meta.get("source") or meta.get("source_file") or "Unknown"
            doc_type = meta.get("type") or ("pageindex" if src.get("source") == "pageindex" else "unknown")
            retrieval_kind = src.get("source", "unknown")  # 'hybrid' | 'pageindex'
            score = src.get("score", 0)

            st.markdown(
                f"**[{i}] {source_name}** `{doc_type}` "
                f"· via `{retrieval_kind}` · score: `{score:.4f}`"
            )
            st.text(src.get("content", "")[:300] + "...")
            st.divider()


# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🎓 University Services RAG Chatbot")
st.caption("Hệ thống hỏi đáp thông tin dịch vụ đại học (Học phí, Học bổng, Ký túc xá, Thư viện)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"])

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/dịch vụ đại học...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu..."):
            try:
                # Bước 1 (Task 9): retrieval — hybrid search + rerank +
                # PageIndex fallback nếu điểm quá thấp
                context_chunks = retrieve(query, top_k=top_k)
            except Exception as e:
                st.error(f"❌ Lỗi ở bước retrieval (Task 9): {e}")
                context_chunks = []

        with st.spinner("Đang tổng hợp câu trả lời..."):
            try:
                # Bước 2 (Task 10): generation có citation, kèm lịch sử hội
                # thoại nếu người dùng bật tuỳ chọn "nhớ ngữ cảnh"
                history = None
                if use_history:
                    # Chỉ lấy role+content, bỏ "sources" (Task 10 không cần,
                    # và API sẽ reject field lạ trong message). Bỏ luôn lượt
                    # user vừa thêm (đã đưa vào user_prompt riêng trong
                    # generate_with_citation, tránh trùng lặp).
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]

                answer_result = generate_with_citation(
                    query, context_chunks=context_chunks, conversation_history=history
                )
                answer = answer_result["answer"]
                # generate_with_citation() trả lại đúng context_chunks đã dùng
                # trong "sources" — dùng lại để đồng bộ (trường hợp hàm tự
                # retrieve nội bộ khi không truyền context_chunks, dù ở đây
                # ta luôn truyền sẵn nên 2 giá trị sẽ giống nhau)
                context_chunks = answer_result.get("sources", context_chunks)
            except (ValueError, RuntimeError) as e:
                answer = f"❌ **Lỗi khi gọi LLM (Task 10):** {e}"
            except Exception as e:
                answer = f"❌ **Lỗi không xác định:** {e}"

        st.markdown(answer)
        if context_chunks:
            _render_sources(context_chunks)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": context_chunks,
    })