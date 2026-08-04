"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search
    2. Merge kết quả bằng RRF (Task 7)
    3. Rerank (mặc định: cross-encoder qua Jina API — xem lý do chọn method
       bên dưới CONFIGURATION)
    4. Nếu top result score (cosine gốc từ semantic_search) < threshold
       → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đã tránh trong code dưới đây:
    Nếu dùng điểm RRF đã fuse để so với score_threshold, sẽ gặp bug thật:
    RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Đặt threshold thấp để "hợp" thang RRF sẽ khiến fallback gần
    như KHÔNG BAO GIỜ trigger — kể cả query hoàn toàn vô nghĩa.

    Cách sửa đúng (đã áp dụng): giữ điểm cosine similarity GỐC của
    dense_results (semantic_search, trước khi qua RRF) làm căn cứ quyết định
    fallback — tách biệt hoàn toàn khỏi điểm RRF/rerank dùng để sắp xếp kết
    quả cuối cùng.
"""

from task5_semantic_search import semantic_search
from task6_lexical_search import lexical_search
from task7_reranking import rerank, rerank_rrf
from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# ⚠️ Giá trị 0.3 dưới đây CHỈ là điểm khởi đầu để test — PHẢI tự calibrate lại
# trước khi dùng thật:
#   1. Chạy vài query CHẮC CHẮN có trong corpus (vd "học phí bao nhiêu") qua
#      semantic_search(), ghi lại score[0] — đây là nhóm "liên quan".
#   2. Chạy vài query CHẮC CHẮN lạc đề (vd "công thức nấu phở", "giá xăng hôm
#      nay") qua semantic_search(), ghi lại score[0] — đây là nhóm "rác".
#   3. Chọn threshold nằm GIỮA khoảng điểm 2 nhóm này. Với bge-m3 (Task 4),
#      quan sát thực tế ở Task 5 cho thấy query liên quan thường >0.55-0.65,
#      nên 0.3 là ngưỡng khá an toàn/rộng rãi — nhưng dữ liệu thật của em có
#      thể khác, đo lại là bắt buộc, không dùng nguyên số này.
SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5

# Method rerank mặc định cho pipeline: "cross_encoder" (Jina API, Task 7) —
# vì đây là method DUY NHẤT trong rerank() không cần thêm tham số ngoài
# (query, candidates, top_k). "mmr" cần query_embedding, "rrf" cần
# ranked_lists — cả 2 không khớp trực tiếp với chữ ký rerank(query,
# candidates, top_k) mà bước 3 của pipeline gọi sau khi đã merge xong.
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | None (bỏ qua rerank)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not query or not query.strip():
        return []

    # --- Bước 1: chạy song song 2 nhánh retrieval ---
    # Lấy top_k*2 ứng viên từ mỗi nhánh để có đủ "nguyên liệu" cho bước
    # merge + rerank thu hẹp lại còn top_k cuối cùng (retrieval rộng trước,
    # rerank thu hẹp sau — chuẩn pattern 2-stage retrieval).
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # --- Bước 4 (tính TRƯỚC, dùng điểm cosine GỐC, chưa qua RRF/rerank) ---
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0

    if best_dense_score < score_threshold:
        print(
            f"  ⚠ Semantic best score ({best_dense_score:.3f}) < threshold "
            f"({score_threshold}) → fallback sang PageIndex"
        )
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback
        # PageIndex cũng không có gì (chưa upload doc, lỗi API, v.v.) →
        # vẫn trả kết quả hybrid thay vì trả về rỗng hoàn toàn, kèm cảnh báo.
        print("  ⚠ PageIndex fallback không có kết quả, trả về hybrid results (chất lượng thấp).")

    # --- Bước 2: merge bằng RRF ---
    if not dense_results and not sparse_results:
        return []

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # --- Bước 3: rerank ---
    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            # rerank_cross_encoder() trả dict mới {**candidate, "score": ...}
            # — giữ nguyên "source": "hybrid" đã gắn ở bước merge (candidate
            # gốc đã có key này nên **candidate tự động mang theo).
        except Exception as e:
            # Rerank qua API ngoài (Jina) có thể lỗi mạng/hết quota — không
            # để lỗi này làm sập cả pipeline, fallback về kết quả merge thô.
            print(f"  ⚠ Rerank lỗi ({e}), dùng kết quả merge (chưa rerank).")
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        if not results:
            print("  (không có kết quả)")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")