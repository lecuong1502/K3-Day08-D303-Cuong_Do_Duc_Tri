"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

=== ĐỒNG BỘ VỚI TASK 4 ===
Module này KHÔNG tự định nghĩa lại chunking — nó import trực tiếp
load_documents() và chunk_documents() từ task4_chunking_indexing.py, để
lexical search (BM25) và semantic search (Task 5) truy hồi trên CÙNG một
tập chunk (cùng chunk_size=500, overlap=50, cùng bước clean_content() lọc
nav-menu rác của news/ và bảng vỡ của legal/ — xem giải thích chi tiết
trong task4_chunking_indexing.py).

Lý do bắt buộc phải đồng bộ:
    1. Nếu BM25 và semantic search chunk khác nhau, không thể kết hợp
       (hybrid search / re-ranking) ở các task sau vì 'chunk' của 2 bên
       không tương ứng 1-1 — không so sánh/merge score được.
    2. Nếu BM25 build trên corpus CHƯA clean, nó sẽ match nhầm các query
       vào chunk rác (menu nav, bảng vỡ) vì BM25 chỉ dựa vào tần suất từ,
       không hiểu ngữ nghĩa để tự loại rác như semantic search có thể làm.
"""

from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from task4_chunking_indexing import chunk_documents, load_documents

# ---------------------------------------------------------------------------
# Không định nghĩa lại CHUNK_SIZE/CHUNK_OVERLAP ở đây — chúng nằm trong
# task4_chunking_indexing.py (CHUNK_SIZE=500, CHUNK_OVERLAP=50) và được áp
# dụng xuyên suốt qua chunk_documents(). Sửa 1 chỗ duy nhất ở Task 4 nếu cần
# đổi thông số, tránh tình trạng 2 module lệch nhau như bản trước.
# ---------------------------------------------------------------------------


def _load_and_chunk_corpus() -> list[dict]:
    """
    Đọc + clean + chunk toàn bộ file .md trong data/standardized/ (output
    Task 3), dùng NGUYÊN pipeline load_documents() -> chunk_documents() của
    Task 4, đảm bảo BM25 và semantic search dùng chung 1 tập chunk.

    Returns:
        List of {'content': str, 'metadata': dict}
        metadata gồm: source (tên file), type ("legal"|"news"), chunk_index
    """
    documents = load_documents()  # đã bao gồm bước clean_content() của Task 4
    if not documents:
        raise FileNotFoundError(
            "Không có document nào trong data/standardized/. Hãy chạy Task 3 trước."
        )

    chunks = chunk_documents(documents)  # dùng chunk_size/overlap của Task 4
    if not chunks:
        raise RuntimeError("chunk_documents() trả về rỗng — kiểm tra lại Task 4.")

    return chunks


# Load + chunk corpus 1 lần khi module được import.
# Lưu ý: nếu data/standardized/ thay đổi (chạy lại Task 3), phải import lại
# module này (hoặc restart Python session) để CORPUS/BM25_INDEX cập nhật.
CORPUS: list[dict] = _load_and_chunk_corpus()  # List of {'content': str, 'metadata': dict}


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    # Tokenize đơn giản: lowercase + split theo khoảng trắng.
    # (Có thể nâng cấp bằng underthesea.word_tokenize cho tiếng Việt để tách
    # đúng từ ghép, ví dụ "học phí" thay vì tách rời "học" và "phí" — nhưng
    # split() đơn giản đã đủ dùng tốt cho BM25 vì BM25 vẫn match theo token
    # đơn lẻ, chỉ là độ chính xác từ ghép sẽ thấp hơn 1 chút.)
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


# Build index 1 lần duy nhất khi import module — tránh phải build lại BM25
# mỗi lần gọi lexical_search() (tốn thời gian nếu corpus lớn).
BM25_INDEX = build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not query or not query.strip():
        return []

    tokenized_query = query.lower().split()
    if not tokenized_query:
        return []

    scores = BM25_INDEX.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    print(f"Corpus: {len(CORPUS)} chunks (đồng bộ với Task 4: chunk_size=500, overlap=50)")

    test_queries = [
        "tuition fee payment methods",
        "học bổng cho sinh viên",
        "accommodation international students",
    ]

    for q in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {q!r}")
        print("=" * 60)
        results = lexical_search(q, top_k=5)
        if not results:
            print("  (không có kết quả)")
        for r in results:
            source = r["metadata"].get("source", "?")
            print(f"[{r['score']:.3f}] ({source}) {r['content'][:100]}...")