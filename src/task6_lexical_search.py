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
"""

from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Cấu hình chunking — dùng CHUNG chunk_size/overlap với Task 4, để lexical
# search và semantic search truy hồi trên cùng một cách chia "đơn vị" văn bản.
# Nếu Task 4 của nhóm bạn dùng thông số khác, sửa 2 hằng số này cho khớp.
# ---------------------------------------------------------------------------
STANDARDIZED_DIR = Path(__file__).resolve().parent.parent / "data" / "standardized"
CHUNK_SIZE = 800     # khớp thông số chính thức của nhóm (Role 1 kiểm tra ở CP2)
CHUNK_OVERLAP = 100  # khớp thông số chính thức của nhóm (Role 1 kiểm tra ở CP2)


def _load_and_chunk_corpus() -> list[dict]:
    """
    Đọc toàn bộ file .md trong data/standardized/ (output Task 3) và chunk
    bằng RecursiveCharacterTextSplitter, trả về list dạng CORPUS yêu cầu.
    """
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {STANDARDIZED_DIR}. Hãy chạy Task 3 (convert markdown) trước."
        )

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"Không có file .md nào trong {STANDARDIZED_DIR}. Hãy chạy Task 3 trước."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    corpus: list[dict] = []
    for filepath in md_files:
        text = filepath.read_text(encoding="utf-8")
        category = filepath.parent.name  # "legal" hoặc "news"
        for i, chunk_text in enumerate(splitter.split_text(text)):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            corpus.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        "source": str(filepath.relative_to(STANDARDIZED_DIR)),
                        "category": category,
                        "chunk_index": i,
                    },
                }
            )
    return corpus


# Load corpus 1 lần khi module được import. Nếu data/standardized/ thay đổi,
# xóa CORPUS/BM25_INDEX rồi import lại (hoặc gọi lại _load_and_chunk_corpus()
# và build_bm25_index() thủ công trong REPL/notebook).
CORPUS: list[dict] = _load_and_chunk_corpus()  # List of {'content': str, 'metadata': dict}


def build_bm25_index(corpus: list[dict]):
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
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


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

    # Get top_k indices
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")