"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/  (lấy API key tại dash.pageindex.ai/api-keys)
SDK & sample code: https://github.com/VectifyAI/PageIndex
Doc SDK chính thức (đối chiếu code này với doc — không đoán từ ví dụ cũ):
    https://docs.pageindex.ai/sdk
    https://docs.pageindex.ai/sdk/legacy/retrieval

PageIndex cho phép RAG mà không cần vector store — sử dụng structural
understanding của document (cây mục lục/section) thay vì embedding.

Lưu ý: endpoint /retrieval hiện đã deprecated (response có field "deprecation"
trỏ sang chat-completion API) nhưng vẫn hoạt động. Ở đây vẫn dùng /retrieval vì
nó trả về node có cấu trúc (title + physical_index) — đúng thứ pipeline cần để
gắn citation, còn chat API chỉ trả câu trả lời dạng văn xuôi.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Nơi lưu PDF trung gian (PageIndex chỉ nhận PDF, không nhận .md trực tiếp
# qua Cloud API) và mapping file -> doc_id để lần sau khỏi phải upload lại.
PDF_CACHE_DIR = Path(__file__).parent.parent / "data" / "_tmp_pdf"
DOC_ID_STORE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

# submit_query() của PageIndex chỉ nhận 1 doc_id/lần (không phải toàn corpus),
# nên phải hỏi từng document đã upload rồi gộp kết quả lại.
#
# Hạn mức này phải PHỦ HẾT corpus. Trước đây để 8 và cắt danh sách doc theo thứ
# tự alphabet -> "legal/tuition-fees-rmit.md" (đứng thứ 9) không bao giờ được
# hỏi, nên câu "tuition fee payment methods" trả về nội dung học bổng thay vì
# biểu phí. Không có điểm số nào để chọn trước doc nào đáng hỏi, nên cắt bớt
# danh sách = mù quáng loại đúng tài liệu chứa câu trả lời.
MAX_DOCS_TO_QUERY = 32

# Các doc được hỏi SONG SONG: mỗi doc tốn 1 vòng submit + poll (vài giây tới
# vài chục giây), hỏi tuần tự cả corpus sẽ mất hàng phút — quá chậm cho một
# nhánh fallback. PageIndex xử lý bất đồng bộ nên phần lớn thời gian là chờ I/O,
# thread pool là đủ (không cần async).
QUERY_WORKERS = 8

# Font Unicode để PDF giữ được dấu tiếng Việt khi render bằng fpdf2 (core font
# Helvetica mặc định chỉ hỗ trợ Latin-1, sẽ thay dấu tiếng Việt bằng "?").
# Dò theo thứ tự: Linux (Ubuntu/Debian, gói fonts-dejavu-core) -> macOS -> Windows.
_UNICODE_FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),   # Ubuntu/Debian
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),             # Fedora/RHEL
    Path("/Library/Fonts/Arial Unicode.ttf"),                   # macOS
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),  # macOS
    Path("C:/Windows/Fonts/arial.ttf"),                         # Windows
    Path("C:/Windows/Fonts/seguiemj.ttf"),                      # Windows fallback
]


def _find_unicode_font() -> Path | None:
    for candidate in _UNICODE_FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _markdown_to_pdf(md_path: Path, pdf_path: Path) -> Path:
    """
    Convert 1 file markdown sang PDF đơn giản để upload lên PageIndex.

    Không cần parse cú pháp markdown (heading, bullet, ...) — PageIndex tự
    phân tích cấu trúc document (structural/tree understanding) từ nội dung
    text + layout PDF, nên ở đây chỉ cần đảm bảo giữ nguyên text gốc.
    """
    from fpdf import FPDF, XPos, YPos
    from fpdf.enums import WrapMode

    text = md_path.read_text(encoding="utf-8")

    pdf = FPDF()
    pdf.add_page()

    font_path = _find_unicode_font()
    if font_path is not None:
        # fpdf2 mới đã BỎ tham số `uni` (add_font tự nhận diện TTF = unicode
        # dựa vào đuôi file .ttf) — truyền uni=True vào bản mới sẽ TypeError.
        pdf.add_font("UnicodeFont", "", str(font_path))
        pdf.set_font("UnicodeFont", size=11)
    else:
        # Không tìm thấy font Unicode nào trên máy — báo lỗi rõ ràng thay vì
        # âm thầm strip dấu tiếng Việt thành "?" (lỗi cũ, rất khó nhận ra khi
        # chỉ nhìn PDF output sơ qua vì layout PDF vẫn "trông ổn").
        raise RuntimeError(
            "Không tìm thấy font Unicode nào trên máy để giữ dấu tiếng Việt trong PDF.\n"
            "Cài đặt (Ubuntu/Debian):  sudo apt install fonts-dejavu-core\n"
            "Sau đó chạy lại script."
        )

    for line in text.splitlines():
        # multi_cell tự wrap + xuống trang mới khi cần, tránh lỗi tràn khổ giấy.
        # Dòng rỗng vẫn cần in 1 space để giữ khoảng cách đoạn văn.
        #
        # new_x=LMARGIN BẮT BUỘC: mặc định của fpdf2 là new_x=XPos.RIGHT, tức
        # con trỏ x nằm lại ở MÉP PHẢI của cell vừa vẽ. Với w=0 (cell kéo dài
        # tới lề phải) thì lần gọi multi_cell kế tiếp có bề rộng khả dụng = 0
        # -> FPDFException("Not enough horizontal space to render a single
        # character") ngay từ dòng thứ hai của file.
        #
        # wrapmode=CHAR: cho phép ngắt dòng GIỮA một "từ". Markdown convert từ
        # PDF gốc đầy dòng mục lục dot-leader ("Tuition Fees .......... 4") và
        # URL dài — đó là các "từ" không có khoảng trắng, dài hơn cả khổ giấy,
        # mà chế độ ngắt theo từ (mặc định) không xử lý được. Ngắt theo ký tự
        # giữ NGUYÊN VẸN text gốc, khác với cách chèn thêm khoảng trắng vào
        # giữa token (làm hỏng URL và sai lệch nội dung PageIndex đọc được).
        pdf.multi_cell(
            0, 6, line if line.strip() else " ",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT, wrapmode=WrapMode.CHAR,
        )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))
    return pdf_path


def upload_documents(force: bool = False):
    """
    Upload toàn bộ markdown documents lên PageIndex (convert sang PDF trước).

    Mặc định BỎ QUA file đã có doc_id trong DOC_ID_STORE — upload lại tốn quota
    trang của PageIndex và bắt phải chờ tree building lại từ đầu. Dùng
    force=True khi nội dung trong data/standardized/ đã thay đổi.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "Chưa set PAGEINDEX_API_KEY trong .env. Đăng ký tại https://pageindex.ai/"
        )

    from pageindex import PageIndexClient  # KHÔNG phải pageindex.client

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    # Key theo đường dẫn TƯƠNG ĐỐI, không phải md_file.name/stem: corpus có
    # 2 thư mục con (legal/ và news/) nên 2 file trùng tên ở 2 nơi sẽ ghi đè
    # PDF của nhau và trùng key trong store.
    existing = {entry["file"]: entry for entry in _load_doc_ids() if entry.get("doc_id")}

    doc_ids = []
    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    for md_file in md_files:
        rel_name = md_file.relative_to(STANDARDIZED_DIR).as_posix()

        if not force and rel_name in existing:
            doc_ids.append(existing[rel_name])
            print(f"  · Bỏ qua (đã upload): {rel_name} -> {existing[rel_name]['doc_id']}")
            continue

        pdf_path = PDF_CACHE_DIR / f"{rel_name.replace('/', '__')}.pdf"
        _markdown_to_pdf(md_file, pdf_path)

        try:
            result = client.submit_document(str(pdf_path))
        except Exception as e:
            # Tài khoản PageIndex free có hạn mức SỐ TRANG. Corpus này ~320
            # trang PDF nên rất dễ đụng {"detail":"LimitReached"} ở giữa chừng.
            # Dừng vòng lặp thay vì ném exception ra ngoài — các doc_id đã
            # upload thành công trước đó vẫn phải được ghi xuống đĩa (xem
            # phần lưu store bên dưới), nếu không sẽ mất sạch và lần chạy sau
            # lại upload lại từ đầu, đốt thêm quota.
            print(f"  ✗ Dừng upload tại {rel_name}: {e}")
            break

        doc_id = result.get("doc_id") or result.get("id")
        doc_ids.append({"file": rel_name, "doc_id": doc_id})
        print(f"  ✓ Uploaded: {rel_name} -> {doc_id} (đang xử lý tree bất đồng bộ...)")

    # Gộp với store cũ để không xoá mất doc_id của file đã upload ở lần chạy
    # trước nhưng lần này bị bỏ dở (vd. hết quota trước khi tới lượt nó).
    merged = dict(existing)
    merged.update({entry["file"]: entry for entry in doc_ids})
    doc_ids = [merged[key] for key in sorted(merged)]

    DOC_ID_STORE.parent.mkdir(parents=True, exist_ok=True)
    DOC_ID_STORE.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Đã lưu {len(doc_ids)} doc_id vào {DOC_ID_STORE}")
    return doc_ids


def rebuild_doc_ids_from_account() -> list[dict]:
    """
    Dựng lại DOC_ID_STORE từ danh sách document ĐANG CÓ trên tài khoản PageIndex.

    Dùng khi store bị mất/xoá nhưng document vẫn nằm trên PageIndex — upload lại
    sẽ đốt quota trang một cách vô ích (tài khoản free rất dễ hết hạn mức).
    Ghép lại bằng tên file: upload_documents() đặt tên PDF là
    "<đường dẫn tương đối với '/' -> '__'>.pdf", nên tách ngược ra được.
    """
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    documents = client.list_documents(limit=100).get("documents", [])

    known = {
        md.relative_to(STANDARDIZED_DIR).as_posix()
        for md in STANDARDIZED_DIR.rglob("*.md")
    }

    recovered = {}
    for doc in documents:
        rel_name = doc.get("name", "").removesuffix(".pdf").replace("__", "/")
        # Chỉ nhận document khớp file có thật trong corpus — tài khoản có thể
        # chứa cả PDF của bài lab/thử nghiệm khác, đưa vào sẽ nhiễu kết quả.
        if rel_name in known:
            recovered[rel_name] = {"file": rel_name, "doc_id": doc["id"]}

    doc_ids = [recovered[key] for key in sorted(recovered)]
    DOC_ID_STORE.parent.mkdir(parents=True, exist_ok=True)
    DOC_ID_STORE.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Khôi phục {len(doc_ids)}/{len(documents)} document vào {DOC_ID_STORE}")
    return doc_ids


def _load_doc_ids() -> list[dict]:
    """Đọc mapping file -> doc_id đã lưu từ lần upload_documents() gần nhất."""
    if not DOC_ID_STORE.exists():
        return []
    return json.loads(DOC_ID_STORE.read_text(encoding="utf-8"))


def _wait_until_ready(client, doc_id: str, timeout: float = 300.0, interval: float = 3.0) -> bool:
    """
    Poll is_retrieval_ready(doc_id) cho tới khi True.

    QUAN TRỌNG: submit_document() chỉ trigger xử lý (tree building chạy
    bất đồng bộ, có thể mất vài chục giây tới vài phút tuỳ độ dài PDF).
    Nếu bỏ qua bước chờ này, submit_query() ngay sau upload sẽ gọi vào
    document CHƯA có tree — trả kết quả rỗng/lỗi mà không báo rõ nguyên nhân.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.is_retrieval_ready(doc_id):
            return True
        time.sleep(interval)
    return False


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


def _parse_physical_index(raw) -> int | None:
    """
    "<physical_index_3>" -> 3.

    PageIndex trả số trang dưới dạng chuỗi có tag chứ không phải int, nên phải
    bóc ra trước khi đưa vào metadata (citation "trang 3" chứ không phải
    "trang <physical_index_3>").
    """
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    match = re.search(r"\d+", str(raw))
    return int(match.group()) if match else None


def _iter_relevant_contents(node: dict):
    """
    Duyệt phẳng "relevant_contents" của 1 retrieved node.

    Schema THẬT (đã verify bằng cách gọi API và dump response, KHÔNG đoán):
        relevant_contents = [
            [ {"section_title": ..., "physical_index": "<physical_index_1>",
               "relevant_content": ...}, ... ],   # <- LỒNG 2 CẤP
            [ {...} ],
        ]
    Mỗi phần tử cấp 1 là 1 LIST, không phải dict — gọi thẳng .get() lên nó sẽ
    ném AttributeError. Hàm này bọc cả trường hợp list phẳng 1 cấp để không vỡ
    nếu API (đang deprecated) đổi schema.
    """
    for group in node.get("relevant_contents") or []:
        if isinstance(group, dict):
            yield group
        elif isinstance(group, list):
            for item in group:
                if isinstance(item, dict):
                    yield item


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

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    def _query_one_doc(entry: dict) -> list[dict]:
        doc_id = entry["doc_id"]

        if not client.is_retrieval_ready(doc_id):
            print(f"  ⚠ Bỏ qua {entry.get('file')}: document chưa xử lý xong tree "
                  f"(is_retrieval_ready=False) — chờ thêm rồi thử lại sau.")
            return []

        try:
            submitted = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = submitted.get("retrieval_id")
            if not retrieval_id:
                # Không có id thì không poll được — báo rõ thay vì để
                # _wait_for_retrieval(None) gọi API với URL /retrieval/None/.
                raise RuntimeError(f"submit_query không trả retrieval_id: {submitted}")
            retrieval = _wait_for_retrieval(client, retrieval_id)
        except Exception as e:
            # 1 document lỗi/timeout không được làm hỏng cả lượt fallback —
            # các document còn lại vẫn có thể chứa câu trả lời.
            print(f"  ⚠ PageIndex query lỗi cho doc {entry.get('file')}: {e}")
            return []

        doc_results = []
        for rank, node in enumerate(retrieval.get("retrieved_nodes") or []):
            node_title = node.get("title")
            for content_item in _iter_relevant_contents(node):
                # PageIndex không trả score số cụ thể — tự gán điểm giảm dần
                # theo thứ hạng node xuất hiện trong response, để tương thích
                # với format score-based của các module retrieval khác.
                score = 1.0 / (rank + 1)
                doc_results.append({
                    # section_title nằm ở content item và cụ thể hơn title của
                    # node (node có thể gộp nhiều section con) — ưu tiên nó.
                    "content": content_item.get("relevant_content", ""),
                    "score": score,
                    "metadata": {
                        "section": content_item.get("section_title") or node_title,
                        "page_index": _parse_physical_index(
                            content_item.get("physical_index")
                        ),
                        "source_file": entry.get("file"),
                        "doc_id": doc_id,
                        "node_id": node.get("id"),
                    },
                    "source": "pageindex",
                })
        return doc_results

    targets = doc_entries[:MAX_DOCS_TO_QUERY]
    results = []
    with ThreadPoolExecutor(max_workers=QUERY_WORKERS) as pool:
        # map() giữ nguyên thứ tự doc đầu vào -> kết quả ổn định giữa các lần
        # chạy, vì điểm số tự gán bị trùng nhau nhiều và sort là stable.
        for doc_results in pool.map(_query_one_doc, targets):
            results.extend(doc_results)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/ (lấy key tại dash.pageindex.ai/api-keys)")
    else:
        print("Uploading documents...")
        upload_documents()

        # Chờ TẤT CẢ doc sẽ được query, không chỉ doc đầu tiên: pageindex_search()
        # bỏ qua mọi doc chưa retrieval_ready, nên nếu chỉ chờ doc[0] thì phần
        # lớn corpus vẫn bị skip và test query trả kết quả nghèo nàn.
        entries = _load_doc_ids()[:MAX_DOCS_TO_QUERY]
        if entries:
            print(f"\nChờ {len(entries)} document sẵn sàng (tree building)...")
            from pageindex import PageIndexClient
            _client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            for _entry in entries:
                _ready = _wait_until_ready(_client, _entry["doc_id"], timeout=300)
                _mark = "✓ Sẵn sàng" if _ready else "⚠ Timeout, vẫn chưa sẵn sàng"
                print(f"  {_mark}: {_entry['file']}")

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        if not results:
            print("  (không có kết quả — có thể document vẫn đang xử lý, thử lại sau vài phút)")
        for r in results:
            _meta = r["metadata"]
            print(f"[{r['score']:.3f}] ({_meta.get('source_file')} "
                  f"— trang {_meta.get('page_index')} — {_meta.get('section')})")
            print(f"    {r['content'][:150]}...")