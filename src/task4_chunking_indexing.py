"""
Task 4 — Chunking & Indexing vào Vector Store.

Pipeline: load markdown -> chunk (RecursiveCharacterTextSplitter) ->
          embed (bge-m3 qua Ollama local) -> index (ChromaDB)

=== CHUNKING STRATEGY ===
Dùng RecursiveCharacterTextSplitter (langchain-text-splitters).
Lý do: corpus gồm 2 loại tài liệu rất khác nhau về cấu trúc —
  - legal/: PDF chính sách dài (vài chục trang), câu văn hành chính dài, nhiều đoạn
    liên tục không heading rõ ràng (vd. đoạn convert từ PDF hai cột, bảng học phí).
  - news/: bài viết ngắn, có heading nhưng không đồng nhất giữa các trang crawl.
  RecursiveCharacterTextSplitter an toàn cho cả 2 loại vì nó thử tách theo cấp độ
  ưu tiên (đoạn -> dòng -> câu -> từ) thay vì phụ thuộc cấu trúc heading cụ thể
  (MarkdownHeaderTextSplitter sẽ tạo chunk rất lệch kích thước nếu 1 vài file
  thiếu heading, ví dụ bảng học phí convert từ PDF).

CHUNK_SIZE = 500 (ký tự):
  - Đủ dài để giữ ngữ cảnh 1 đoạn chính sách hoàn chỉnh (vd. 1 điều khoản refund,
    1 mục điều kiện học bổng) nhưng không quá dài để loãng embedding vector
    (embedding của đoạn quá dài sẽ mất độ đặc trưng, ảnh hưởng đến độ chính xác
    retrieval ở Task 5).
  - 500 ký tự tiếng Việt/Anh ~ 100-130 tokens, phù hợp với context ngắn cho
    retrieval-augmented QA (không cần chunk to như summarization task).

CHUNK_OVERLAP = 50 (10% của chunk_size):
  - Tránh cắt đứt câu/ý ở ranh giới chunk (vd. câu điều kiện học bổng bị cắt
    ngay giữa "phải có GPA >= 3.0 VÀ hoàn thành..." — nếu không overlap, chunk
    sau sẽ mất phần đầu điều kiện).
  - 10% là mức overlap tiêu chuẩn khuyến nghị bởi LangChain docs, cân bằng giữa
    tránh mất ngữ cảnh và tránh trùng lặp dữ liệu quá nhiều (overlap cao hơn
    làm tăng số chunk, tăng chi phí embedding mà lợi ích giảm dần).

=== EMBEDDING MODEL ===
Model: bge-m3:567m — chạy LOCAL qua Ollama (không qua HuggingFace/sentence-transformers).
  docker exec -it ollama ollama list  →  đã có sẵn bge-m3:567m (1.2GB)

Lý do chọn Ollama thay vì sentence-transformers:
  - Model đã pull sẵn trong container Ollama, không cần tải lại qua HF Hub
    (tiết kiệm băng thông, tránh lỗi mạng khi tải model từ HF trong môi trường
    company/university có thể bị chặn/giới hạn).
  - Không cần cài torch + sentence-transformers (nặng, nhiều dependency), chỉ
    cần `requests` gọi REST API local — nhẹ hơn nhiều cho máy cấu hình yếu.
  - bge-m3 là multilingual model, hỗ trợ tốt cả tiếng Việt lẫn tiếng Anh — phù
    hợp vì corpus có cả file tiếng Anh (RMIT policies gốc tiếng Anh) và có thể
    truy vấn bằng tiếng Việt ở Task 5.

Embedding dimension: 1024 (bge-m3 chuẩn, kể cả bản quantize 567m vẫn giữ dim gốc).

Cài đặt / yêu cầu chạy trước:
    # 1. Đảm bảo Ollama server đang chạy và model đã pull:
    docker exec -it ollama ollama list
    #   NAME              SIZE      → phải thấy bge-m3:567m

    # 2. Cài Python deps (không cần sentence-transformers/torch):
    pip install langchain-text-splitters chromadb requests

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from pathlib import Path

import requests

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # RecursiveCharacterTextSplitter

# Ollama local — đổi OLLAMA_HOST nếu server chạy nơi khác (vd container tên khác,
# hoặc máy host thay vì localhost khi script chạy trong container riêng)
OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3:567m"
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            print(f"  ⚠ Bỏ qua file rỗng: {md_file.name}")
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_text(text: str) -> list[float]:
    """Gọi Ollama REST API /api/embed để lấy embedding của 1 đoạn text."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    # /api/embed trả về {"embeddings": [[...]]} (list lồng, kể cả 1 input)
    return data["embeddings"][0]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng bge-m3 qua Ollama local.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    # Kiểm tra Ollama server sống trước khi chạy cả loop, tránh fail giữa chừng
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Không kết nối được Ollama tại {OLLAMA_HOST}. "
            f"Kiểm tra: docker ps | grep ollama  (container có đang chạy không?). "
            f"Lỗi gốc: {e}"
        )

    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        chunk["embedding"] = embed_text(chunk["content"])
        if i % 10 == 0 or i == total:
            print(f"  Embedded {i}/{total} chunks...")
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB (persistent, local, không cần Docker).
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Xóa collection cũ nếu tồn tại để tránh trộn lẫn dữ liệu cũ/mới (xem lưu ý ở đầu file)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  ⚠ Đã xóa collection cũ '{COLLECTION_NAME}' trước khi reindex")
    except Exception:
        pass  # collection chưa tồn tại lần đầu chạy — bỏ qua

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return collection


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} via Ollama @ {OLLAMA_HOST} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    if not docs:
        print("⚠ Không tìm thấy file .md nào trong data/standardized/. Chạy Task 3 trước.")
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    print("\nĐang embed qua Ollama (có thể mất vài phút tuỳ số lượng chunk)...")
    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    collection = index_to_vectorstore(chunks)
    print(f"✓ Indexed {collection.count()} chunks vào ChromaDB tại {CHROMA_DIR}")


if __name__ == "__main__":
    run_pipeline()