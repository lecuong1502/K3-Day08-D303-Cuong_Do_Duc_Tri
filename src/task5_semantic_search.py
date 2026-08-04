"""
Task 5 — Semantic Search Module.

Tìm kiếm ngữ nghĩa (dense retrieval) trên vector store đã index ở Task 4.
Dùng LẠI đúng embedding model + Ollama client + ChromaDB collection config
từ Task 4 để đảm bảo query vector và index vector nằm cùng không gian
(embed query bằng model khác với model lúc index sẽ cho kết quả sai lệch
hoàn toàn dù không báo lỗi).

Yêu cầu file này nằm cùng thư mục src/ với task4_chunking_indexing.py.
"""

from pathlib import Path

import chromadb

# Import lại config + hàm embed từ Task 4 — không định nghĩa lại để tránh
# lệch cấu hình (vd đổi EMBEDDING_MODEL ở Task 4 mà quên sửa ở đây).
from task4_chunking_indexing import (
    CHROMA_DIR,
    COLLECTION_NAME,
    embed_text,
)

_collection = None  # cache collection, tránh mở lại PersistentClient mỗi lần gọi search


def _get_collection():
    """Mở (hoặc lấy cache) ChromaDB collection đã index ở Task 4."""
    global _collection
    if _collection is None:
        if not Path(CHROMA_DIR).exists():
            raise FileNotFoundError(
                f"Không tìm thấy {CHROMA_DIR}. Chạy Task 4 (index) trước khi search."
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Không tìm thấy collection '{COLLECTION_NAME}' trong {CHROMA_DIR}. "
                f"Chạy lại Task 4 để index dữ liệu. Lỗi gốc: {e}"
            )
    return _collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score (càng cao càng liên quan)
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending.
    """
    if not query or not query.strip():
        return []

    # Bước 1: Embed query bằng đúng model + endpoint Ollama đã dùng ở Task 4
    query_vector = embed_text(query.strip())
    if query_vector is None:
        raise RuntimeError(
            "Không embed được query (Ollama trả lỗi). Kiểm tra server Ollama đang chạy."
        )

    # Bước 2: Query ChromaDB (collection đã tạo với hnsw:space="cosine" ở Task 4)
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Bước 3: Chroma trả cosine DISTANCE (0 = giống hệt, 2 = đối lập hoàn toàn)
    # → convert sang similarity score dễ đọc hơn: score = 1 - distance
    # (khoảng giá trị điển hình 0..1 cho các cặp câu liên quan về ngữ nghĩa)
    output = []
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results["distances"] else []

    for doc, meta, dist in zip(docs, metas, dists):
        score = max(0.0, 1.0 - dist)
        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": meta,
        })

    # ChromaDB đã trả về theo thứ tự gần nhất trước, nhưng sort lại tường minh
    # để đảm bảo đúng contract của hàm (yêu cầu đề bài: "sorted descending")
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    test_queries = [
        "what is the tuition fee",
        "học bổng cho sinh viên có GPA cao",
        "accommodation for international students",
    ]

    for q in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {q!r}")
        print("=" * 60)
        results = semantic_search(q, top_k=5)
        if not results:
            print("  (không có kết quả)")
        for r in results:
            source = r["metadata"].get("source", "?")
            print(f"[{r['score']:.4f}] ({source}) {r['content'][:100]}...")