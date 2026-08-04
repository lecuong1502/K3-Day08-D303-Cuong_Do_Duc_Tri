"""
Task 10 — Generation Có Citation.

Sắp xếp lại context chunks sau reranking để tránh "lost in the middle",
inject vào prompt, và yêu cầu LLM trả lời có citation.

LLM: gọi qua OpenRouter API (https://openrouter.ai/), model openai/gpt-4o.

Cài đặt:
    pip install requests python-dotenv

Thêm vào .env (cùng thư mục với JINA_API_KEY, PAGEINDEX_API_KEY):
    OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxx

Lấy API key tại: https://openrouter.ai/keys
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-4o")

SYSTEM_PROMPT = """Answer the following question comprehensively.
For every statement of fact or claim, immediately insert a citation
in brackets linking to the specific source
(e.g., [Author/Platform Name, Year]).
If the information is not explicitly stated in the provided context
or knowledge base, state 'I cannot verify this information'
rather than guessing."""


# =============================================================================
# Document reordering — tránh "lost in the middle"
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks theo pattern: quan trọng nhất ở đầu và cuối,
    ít quan trọng hơn ở giữa.

    LÝ DO: nghiên cứu "Lost in the Middle" (Liu et al., 2023) cho thấy LLM
    nhớ/dùng tốt nhất thông tin ở ĐẦU và CUỐI context window, kém nhất ở
    GIỮA — bất kể model có context window dài bao nhiêu. Nếu giữ nguyên thứ
    tự relevance giảm dần [1,2,3,4,5] (chunk liên quan nhất ở đầu), các
    chunk hạng 2-3 (vẫn khá quan trọng) lại rơi đúng vào "vùng chết" giữa
    context. Đảo thành [1,3,5,4,2]: chunk quan trọng nhất (1) vẫn ở đầu (vị
    trí nhớ tốt nhất), chunk quan trọng NHÌ (2) được đẩy ra CUỐI (vị trí nhớ
    tốt thứ nhì), các chunk kém quan trọng hơn (3,4,5) bị dồn vào giữa — nơi
    chúng "mất" cũng không ảnh hưởng nhiều vì vốn dĩ ít quan trọng.

    Thuật toán: 2 con trỏ left/right, duyệt chunks theo thứ tự relevance
    giảm dần (input phải ĐÃ sort desc — đúng như output của rerank()/
    retrieve()), luân phiên gán vào đầu rồi cuối:
        rank1 → vị trí đầu (left++)
        rank2 → vị trí cuối (right--)
        rank3 → vị trí đầu tiếp theo (left++)
        rank4 → vị trí cuối tiếp theo (right--)
        ...
    Ví dụ: input rank-order [1,2,3,4,5] → output [1,3,5,4,2] (khớp đúng ví
    dụ trong đề bài).

    Args:
        chunks: List chunks đã sort theo relevance giảm dần (vd output của
            task9_retrieval_pipeline.retrieve())

    Returns:
        List chunks cùng nội dung, thứ tự đã đảo theo pattern trên.
    """
    n = len(chunks)
    if n <= 2:
        return chunks  # không đủ để tạo hiệu ứng "giữa", giữ nguyên

    result: list[dict | None] = [None] * n
    left, right = 0, n - 1

    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            result[left] = chunk
            left += 1
        else:
            result[right] = chunk
            right -= 1

    return result


# =============================================================================
# Context formatting
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format context chunks kèm số thứ tự + metadata nguồn, để LLM có thể trích
    dẫn đúng bằng cách tham chiếu lại số/tên nguồn này trong câu trả lời.

    Format mỗi block: "[N] (Nguồn: <source>) <nội dung>" — LLM được yêu cầu
    (qua SYSTEM_PROMPT) trích dẫn dạng [Nguồn, Năm]; vì metadata của corpus
    hiện KHÔNG có field năm xuất bản tường minh (RMIT không ghi rõ năm ban
    hành trên từng trang), dùng tên file/nguồn làm định danh trích dẫn — nếu
    metadata có key 'year' (vd bổ sung được từ tên file "student-fees...2026"),
    hàm sẽ tự thêm vào citation.
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source") or meta.get("source_file") or meta.get("section") or "unknown"
        year = meta.get("year")
        citation_label = f"{source}, {year}" if year else str(source)

        lines.append(f"[{i}] (Nguồn: {citation_label})\n{chunk['content']}")

    return "\n\n---\n\n".join(lines)


# =============================================================================
# Generation
# =============================================================================

def generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
    top_k: int = 5,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Sinh câu trả lời có citation từ context đã retrieve.

    Steps:
        0. (Nếu không truyền context_chunks) Tự gọi retrieve() (Task 9) để
           lấy context — cho phép dùng hàm này độc lập chỉ với `query`,
           không bắt buộc caller phải tự chạy Task 9 trước.
        1. Reorder chunks để tránh lost in the middle
        2. Format context với source metadata
        3. Inject vào prompt với SYSTEM_PROMPT (+ lịch sử hội thoại nếu có)
        4. Gọi LLM (OpenRouter — gpt-4o)
        5. Return dict {"answer": str, "sources": list[dict]}

    Args:
        query: Câu hỏi gốc của người dùng
        context_chunks: List chunks ĐÃ retrieve sẵn (vd từ app.py, đã có kết
            quả Task 9). Nếu để None, hàm tự gọi retrieve(query, top_k=top_k)
            — tiện cho việc test/dùng độc lập module này.
        top_k: Chỉ dùng khi context_chunks=None (số chunks tự retrieve)
        conversation_history: List các lượt hội thoại TRƯỚC ĐÓ, dạng
            [{"role": "user"|"assistant", "content": str}, ...] — dùng để hỗ
            trợ follow-up questions. LƯU Ý: history chỉ ảnh hưởng cách LLM
            DIỄN GIẢI câu hỏi khi trả lời — nếu context_chunks=None, bước
            retrieve() nội bộ vẫn chỉ dùng `query` hiện tại (KHÔNG tự mở
            rộng câu truy vấn bằng ngữ cảnh hội thoại). Giới hạn đã biết,
            không phải bug.

    Returns:
        {
            "answer": str,       # có citation [Nguồn, Năm], hoặc
                                  # "I cannot verify this information."
            "sources": list[dict]  # context_chunks đã dùng (rỗng nếu
                                    # không tìm được context nào)
        }
    """
    if context_chunks is None:
        # Lazy import — tránh việc import task10 kéo theo toàn bộ chuỗi
        # import của task9 (task5-8) trong trường hợp caller đã tự có sẵn
        # context_chunks và không cần dùng tới retrieve() ở đây.
        from task9_retrieval_pipeline import retrieve
        context_chunks = retrieve(query, top_k=top_k)

    if not context_chunks:
        return {"answer": "I cannot verify this information.", "sources": []}

    ordered_chunks = reorder_for_llm(context_chunks)
    context_text = format_context(ordered_chunks)

    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using ONLY the context above. Cite sources "
        f"using the format [Nguồn, Năm] (or [Nguồn] if no year is given) "
        f"immediately after each claim."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        # Chỉ giữ vài lượt gần nhất, tránh context quá dài (tốn token/cost
        # không cần thiết cho model trả phí như gpt-4o qua OpenRouter).
        messages.extend(conversation_history[-6:])
    messages.append({"role": "user", "content": user_prompt})

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY chưa được set trong .env. "
            "Lấy key tại https://openrouter.ai/keys rồi thêm dòng:\n"
            "  OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxx"
        )

    # --- Sampling params cho LLM sinh câu trả lời ---
    # temperature=0.2: RAG QA cần bám sát context, KHÔNG cần sáng tạo — nhiệt
    #   độ thấp giảm mạnh khả năng model "phịa" thêm chi tiết không có trong
    #   context (hallucination), đánh đổi lấy văn phong hơi khô hơn (chấp
    #   nhận được cho use-case tra cứu chính sách/hỏi đáp thông tin).
    # top_p=0.9: nucleus sampling vẫn giữ 1 chút đa dạng từ vựng (câu trả lời
    #   không bị lặp từ máy móc như temperature=0 tuyệt đối), nhưng cắt bỏ
    #   phần đuôi phân phối xác suất thấp — hạn chế model chọn từ/token hiếm,
    #   lạc chủ đề.
    # (GPT-4o qua OpenRouter không expose tham số top_k riêng như Ollama/local
    #   model — chỉ hỗ trợ temperature + top_p theo chuẩn OpenAI API.)
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GENERATION_MODEL,
                "messages": messages,
                "temperature": 0.2,
                "top_p": 0.9,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        raise RuntimeError(f"OpenRouter API lỗi: {e} — chi tiết: {body}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Không gọi được OpenRouter API (kiểm tra mạng): {e}")

    data = response.json()
    answer_text = data["choices"][0]["message"]["content"].strip()
    return {"answer": answer_text, "sources": context_chunks}


if __name__ == "__main__":
    # Test độc lập với dummy chunks (không phụ thuộc Task 9 chạy được hay
    # không — dễ debug riêng phần generation/reordering).
    dummy_chunks = [
        {
            "content": "Tuition fees at RMIT Vietnam are paid each semester on a course-by-course basis.",
            "score": 0.85,
            "metadata": {"source": "tuition-fees-rmit.md"},
        },
        {
            "content": "Payment methods include online payment via myRMIT, bank transfer, and campus cashier.",
            "score": 0.78,
            "metadata": {"source": "tuition-fees-rmit.md"},
        },
        {
            "content": "Scholarship applicants must have a GPA of at least 3.0/4.0.",
            "score": 0.60,
            "metadata": {"source": "academic-achievement-scholarship-rmit.md"},
        },
        {
            "content": "The library is open from 8am to 9pm on weekdays.",
            "score": 0.30,
            "metadata": {"source": "rmit-vietnam-factsheet-rmit.md"},
        },
        {
            "content": "International students are eligible for medical insurance coverage.",
            "score": 0.55,
            "metadata": {"source": "accommodation-services-rmit.md"},
        },
    ]

    print("=== Test reorder_for_llm ===")
    ordered = reorder_for_llm(dummy_chunks)
    for c in ordered:
        print(f"  [{c['score']:.2f}] {c['content'][:60]}...")

    print("\n=== Test generate_with_citation ===")
    try:
        result = generate_with_citation("What are the tuition payment methods?", dummy_chunks)
        print("Answer:", result["answer"])
        print(f"Sources: {len(result['sources'])} chunks")
    except (ValueError, RuntimeError) as e:
        print(f"⚠ {e}")