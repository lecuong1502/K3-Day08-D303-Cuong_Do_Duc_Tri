"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Nơi lưu PDF trung gian (PageIndex chỉ nhận PDF, không nhận .md trực tiếp)
# và mapping file -> doc_id để lần sau khỏi phải upload lại (đã có trong .gitignore).
PDF_CACHE_DIR = Path(__file__).parent.parent / "data" / "_tmp_pdf"
DOC_ID_STORE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

# Font Unicode có sẵn trên Windows, dùng để PDF giữ được dấu tiếng Việt khi render
# bằng fpdf2 (core font Helvetica mặc định chỉ hỗ trợ latin-1).
_WINDOWS_UNICODE_FONT = Path("C:/Windows/Fonts/arial.ttf")


def _markdown_to_pdf(md_path: Path, pdf_path: Path) -> Path:
    """
    Convert 1 file markdown sang PDF đơn giản để upload lên PageIndex.

    Không cần parse cú pháp markdown (heading, bullet, ...) — PageIndex tự
    phân tích cấu trúc document (structural/tree understanding) từ nội dung
    text + layout PDF, nên ở đây chỉ cần đảm bảo giữ nguyên text gốc.
    """
    from fpdf import FPDF

    text = md_path.read_text(encoding="utf-8")

    pdf = FPDF()
    pdf.add_page()

    if _WINDOWS_UNICODE_FONT.exists():
        pdf.add_font("Arial", "", str(_WINDOWS_UNICODE_FONT), uni=True)
        pdf.set_font("Arial", size=11)
    else:
        # Fallback khi không chạy trên Windows (vd. CI/Linux): dùng core font,
        # ký tự ngoài latin-1 (dấu tiếng Việt) sẽ bị thay bằng "?" thay vì crash.
        pdf.set_font("Helvetica", size=11)
        text = text.encode("latin-1", errors="replace").decode("latin-1")

    for line in text.splitlines():
        # multi_cell tự wrap + xuống trang mới khi cần, tránh lỗi tràn khổ giấy.
        # Dòng rỗng vẫn cần in 1 space để giữ khoảng cách đoạn văn.
        pdf.multi_cell(0, 6, line if line.strip() else " ")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))
    return pdf_path


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "Chưa set PAGEINDEX_API_KEY trong .env. Đăng ký tại https://pageindex.ai/"
        )

    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    doc_ids = []
    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    for md_file in md_files:
        pdf_path = PDF_CACHE_DIR / f"{md_file.stem}.pdf"
        _markdown_to_pdf(md_file, pdf_path)

        resp = client.submit_document(str(pdf_path))
        doc_id = resp.get("doc_id") or resp.get("id")
        doc_ids.append({"file": md_file.name, "doc_id": doc_id})
        print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")

    # Lưu mapping file -> doc_id ra đĩa để pageindex_search() tái sử dụng, vì
    # tree generation + OCR của PageIndex chạy async và tốn thời gian — không
    # muốn upload lại toàn bộ corpus mỗi lần gọi search.
    DOC_ID_STORE.parent.mkdir(parents=True, exist_ok=True)
    DOC_ID_STORE.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Đã lưu {len(doc_ids)} doc_id vào {DOC_ID_STORE}")

    return doc_ids


def _load_doc_ids() -> list[dict]:
    """Đọc mapping file -> doc_id đã lưu từ lần upload_documents() gần nhất."""
    if not DOC_ID_STORE.exists():
        return []
    return json.loads(DOC_ID_STORE.read_text(encoding="utf-8"))


def _wait_for_retrieval(client, retrieval_id: str, timeout: float = 60.0, interval: float = 2.0) -> dict:
    """
    Poll GET /retrieval/{id} cho tới khi status == "completed".

    submit_query() chỉ submit job và trả về retrieval_id ngay lập tức — kết quả
    thật phải poll riêng vì PageIndex xử lý bất đồng bộ.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = client.get_retrieval(retrieval_id)
        status = result.get("status")
        if status == "completed":
            return result
        if status == "failed":
            raise RuntimeError(f"PageIndex retrieval thất bại: {result}")
        time.sleep(interval)
    raise TimeoutError(f"PageIndex retrieval {retrieval_id} không xong sau {timeout}s")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "Chưa set PAGEINDEX_API_KEY trong .env. Đăng ký tại https://pageindex.ai/"
        )

    doc_entries = _load_doc_ids()
    if not doc_entries:
        raise RuntimeError(
            "Chưa có document nào trên PageIndex — chạy upload_documents() trước."
        )

    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    # submit_query() của PageIndex chỉ nhận 1 doc_id/lần (không phải toàn corpus),
    # nên phải hỏi lần lượt từng document đã upload rồi gộp kết quả lại. Giới hạn
    # số doc để tránh quá nhiều API call/độ trễ khi dùng làm fallback (chỉ trigger
    # khi hybrid search điểm thấp, nên vẫn cần phản hồi tương đối nhanh).
    MAX_DOCS_TO_QUERY = 8
    results = []
    for entry in doc_entries[:MAX_DOCS_TO_QUERY]:
        doc_id = entry["doc_id"]
        try:
            submitted = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
            retrieval = _wait_for_retrieval(client, retrieval_id)
        except Exception as e:
            print(f"  ⚠ PageIndex query lỗi cho doc {entry.get('file')}: {e}")
            continue

        # Schema thực tế (verify bằng json.dumps(retrieval) trước khi parse, theo
        # cảnh báo deprecation ở đầu file, KHÔNG đoán từ code mẫu cũ):
        #   retrieval["retrieved_nodes"] = [
        #       {..., "relevant_contents": [[{"section_title": ..., "relevant_content": ...}, ...]]},
        #       ...
        #   ]
        for node in retrieval.get("retrieved_nodes", [])[:2]:
            for group in node.get("relevant_contents", []):
                for rank, item in enumerate(group):
                    # PageIndex không trả về score số cụ thể — tự gán điểm giảm dần
                    # theo thứ hạng xuất hiện trong response, để tương thích với các
                    # module retrieval khác trong pipeline (đều sort theo score desc).
                    score = 1.0 / (rank + 1)
                    results.append({
                        "content": item.get("relevant_content", ""),
                        "score": score,
                        "metadata": {
                            "section": item.get("section_title"),
                            "source_file": entry.get("file"),
                            "doc_id": doc_id,
                        },
                        "source": "pageindex",
                    })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
