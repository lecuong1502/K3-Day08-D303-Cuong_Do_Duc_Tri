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

import json
import time
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

def clean_content(text: str, doc_type: str) -> str:
    """
    Loại bỏ rác trước khi chunk — quan sát từ dữ liệu thật cho thấy 2 loại rác
    khác nhau giữa legal/ và news/:

    - news/ (crawl4ai): mỗi file chứa nguyên khối HTML nav menu của RMIT
      (header dropdown, sidebar, footer — hàng trăm dòng dạng "* [Text](url)")
      dài hơn nội dung bài viết thật rất nhiều lần. Nếu không lọc, đa số chunk
      sẽ là menu link thay vì nội dung, làm loãng retrieval ở Task 5.
    - legal/ (PDF convert): các trang mục lục/bảng ranking bị MarkItDown parse
      vỡ thành bảng markdown rác (nhiều dòng "| --- | --- |" xen kẽ text bị
      trộn cột), không mang giá trị ngữ nghĩa liền mạch.
    """
    lines = text.split("\n")
    cleaned = []

    # Dòng chỉ chứa 1 markdown link (có thể có bullet đầu dòng) → nav/footer link
    nav_link_only = re.compile(r"^\s*[\*\-]?\s*\[[^\]]*\]\([^\)]*\)\s*$")
    # Dòng rác kiểu "[](javascript:void(0);)", "Search field", v.v. hay gặp trong crawl RMIT
    ui_junk = re.compile(
        r"^\s*(\[\]\(javascript:void[^\)]*\)|Search( field)?|"
        r"\[SKIP TO CONTENT\].*|\[\]\(https://www\.rmit\.edu\.vn/?\))\s*$"
    )
    # Dòng bảng markdown chỉ toàn dấu phân cách (table separator rác)
    table_separator_only = re.compile(r"^\s*\|[\s\-\|:]+\|\s*$")

    for line in lines:
        if doc_type == "news" and (nav_link_only.match(line) or ui_junk.match(line)):
            continue
        if doc_type == "legal" and table_separator_only.match(line):
            continue
        cleaned.append(line)

    result = "\n".join(cleaned)
    # gộp nhiều dòng trống liên tiếp thành 1 (do vừa xóa nhiều dòng)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/, sau đó làm sạch nội dung
    (xem clean_content()) trước khi trả về để chunk_documents() không bị nhiễu
    bởi nav-menu (news) hay bảng vỡ (legal).

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        raw = md_file.read_text(encoding="utf-8").strip()
        if not raw:
            print(f"  ⚠ Bỏ qua file rỗng: {md_file.name}")
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"

        content = clean_content(raw, doc_type)
        if len(content) < 50:
            print(f"  ⚠ Bỏ qua file gần như rỗng sau khi clean: {md_file.name}")
            continue

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
    skipped = 0
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if not _is_valid_chunk(chunk_text):
                skipped += 1
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    if skipped:
        print(f"  ⚠ Đã lọc bỏ {skipped} chunk rác (rỗng/chỉ chứa ký tự markdown thừa)")
    return chunks


CHECKPOINT_FILE = Path(__file__).parent.parent / "data" / "embed_checkpoint.json"


import re


def _is_valid_chunk(text: str) -> bool:
    """
    Lọc chunk rác trước khi gửi embed — nguyên nhân phổ biến gây lỗi
    'unsupported value: NaN' từ Ollama là chunk gần như trống hoặc chỉ chứa
    ký tự markdown thừa (vd separator bảng "|---|---|", dòng toàn dấu gạch),
    khiến model không sinh được vector hợp lệ.
    """
    stripped = text.strip()
    if len(stripped) < 10:
        return False
    # còn lại sau khi bỏ khoảng trắng/markdown table chars/dấu câu phải có
    # ít nhất vài ký tự chữ/số thật sự
    alnum_count = len(re.sub(r"[\s\|\-:=_*#>`.,]", "", stripped))
    return alnum_count >= 8


def embed_text(text: str, max_retries: int = 3) -> list[float] | None:
    """
    Gọi Ollama REST API /api/embed để lấy embedding của 1 đoạn text.
    Retry với backoff cho lỗi mạng/quá tải tạm thời. Lỗi NaN là lỗi xác định
    (chunk cụ thể luôn gây lỗi, không phải do server quá tải) nên không retry —
    trả về None để chunk đó bị bỏ qua thay vì làm crash cả pipeline.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": text},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"][0]
        except requests.exceptions.HTTPError as e:
            last_error = e
            body = ""
            try:
                body = e.response.text[:300]
            except Exception:
                pass

            if "NaN" in body or "unsupported value" in body:
                # lỗi xác định do nội dung chunk — retry vô ích, bỏ qua ngay
                print(f"    ✗ Chunk gây lỗi NaN (bỏ qua, không index): {text[:80]!r}...")
                return None

            print(f"    ⚠ Lỗi HTTP (lần {attempt}/{max_retries}): {e} — Ollama trả về: {body}")
            time.sleep(2 * attempt)
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"    ⚠ Lỗi kết nối (lần {attempt}/{max_retries}): {e}")
            time.sleep(2 * attempt)

    print(f"    ✗ Bỏ qua chunk sau {max_retries} lần thử lỗi: {last_error}")
    return None


def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return {}


def _save_checkpoint(cache: dict):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(cache), encoding="utf-8")


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng bge-m3 qua Ollama local.
    Có checkpoint: nếu script bị lỗi/dừng giữa chừng, chạy lại sẽ load embedding
    đã tính từ lần trước (khớp theo source+chunk_index) thay vì embed lại từ đầu —
    quan trọng vì corpus có ~1900 chunks, embed lại từ đầu tốn nhiều phút.

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

    cache = _load_checkpoint()
    if cache:
        print(f"  ↻ Tìm thấy checkpoint với {len(cache)} embedding đã tính từ lần chạy trước")

    total = len(chunks)
    valid_chunks = []
    try:
        for i, chunk in enumerate(chunks, 1):
            key = f"{chunk['metadata']['source']}_chunk_{chunk['metadata']['chunk_index']}"
            if key in cache:
                emb = cache[key]
            else:
                emb = embed_text(chunk["content"])
                if emb is not None:
                    cache[key] = emb

            if emb is not None:
                chunk["embedding"] = emb
                valid_chunks.append(chunk)
            # emb is None → chunk bị bỏ qua vĩnh viễn (lỗi NaN xác định), không index

            if i % 10 == 0 or i == total:
                print(f"  Embedded {i}/{total} chunks... ({len(valid_chunks)} hợp lệ)")
                _save_checkpoint(cache)
    except Exception:
        _save_checkpoint(cache)
        print(f"  ⚠ Đã lưu checkpoint tại {CHECKPOINT_FILE} — chạy lại script sẽ resume từ đây.")
        raise

    _save_checkpoint(cache)
    return valid_chunks


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