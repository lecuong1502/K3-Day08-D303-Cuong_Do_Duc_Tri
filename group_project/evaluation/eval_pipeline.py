"""
RAG Evaluation Pipeline — DeepEval.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Cài đặt:
    pip install deepeval pandas python-dotenv

⚠️ RATE LIMIT — ĐỌC KỸ TRƯỚC KHI CHẠY FULL DATASET:
    DeepEval gọi LLM giám khảo RẤT NHIỀU LẦN, không phải 1 lần/câu hỏi.
    FaithfulnessMetric tách answer thành từng claim rồi verify từng claim
    (nhiều call), AnswerRelevancyMetric/ContextualRecall/ContextualPrecision
    cũng tương tự → 1 câu hỏi có thể tốn 5-15 lệnh gọi LLM CHO MỖI METRIC.
    Với 15 câu x 2 configs x 4 metrics, tổng số lệnh gọi LLM giám khảo có
    thể lên tới VÀI TRĂM.

    Nếu dùng model OpenRouter dạng ":free" (vd "meta-llama/llama-3.1-8b-
    instruct:free"): giới hạn 50 request/ngày CHO CẢ TÀI KHOẢN (không phải
    theo model hay theo key — đổi model free khác/tạo key mới KHÔNG reset
    quota). → Đặt MAX_QUESTIONS nhỏ (vd 5) bên dưới để chạy kịp, hoặc dùng
    model trả phí rẻ (khuyến nghị: "openai/gpt-4o-mini") để có quota cao hơn
    hẳn (không bị giới hạn 50/ngày).

    Script này được thiết kế để SỐNG SÓT khi bị rate-limit giữa chừng: chấm
    điểm TỪNG metric riêng lẻ (không dùng deepeval.evaluate() batch 1 lần
    cho cả dataset), nên nếu lỗi ở câu hỏi thứ 8, kết quả của 7 câu trước đó
    vẫn được lưu vào results.md thay vì mất trắng toàn bộ.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from task5_semantic_search import semantic_search  # noqa: E402
from task9_retrieval_pipeline import retrieve  # noqa: E402
from task10_generation import generate_with_citation as _generate_with_citation  # noqa: E402

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
RAW_RESULTS_PATH = Path(__file__).parent / "eval_results_raw.csv"

# ⚠️ Đặt None để chạy full golden dataset. Nếu dùng model OpenRouter ":free"
# (xem cảnh báo rate-limit ở đầu file), đặt số nhỏ (vd 5) để chạy kịp trong
# quota 50 request/ngày.
MAX_QUESTIONS = None

# Model LÀM GIÁM KHẢO chấm điểm — nên KHÁC với GENERATION_MODEL ở Task 10
# (tránh self-preference bias: model tự chấm cao cho câu trả lời chính nó
# sinh ra). Mặc định dùng bản mini rẻ để tiết kiệm quota/chi phí vì phải gọi
# rất nhiều lần (xem cảnh báo ở đầu file).
EVAL_MODEL = os.getenv("EVAL_MODEL", "openai/gpt-4o-mini")


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if MAX_QUESTIONS:
        data = data[:MAX_QUESTIONS]
    return data


# =============================================================================
# RAG Pipeline adapter — bọc các hàm task5/task9/task10 thành object có
# .generate_with_citation(question), hỗ trợ đổi "config" retrieval để phục
# vụ A/B comparison (compare_configs bên dưới).
# =============================================================================

class RAGPipeline:
    """
    Adapter mỏng quanh Task 9 (retrieve) + Task 10 (generate_with_citation).

    config:
        "hybrid_rerank" — retrieve() đầy đủ (Task 9): hybrid RRF + cross-
            encoder rerank + PageIndex fallback nếu điểm thấp
        "dense_only"    — chỉ semantic_search() (Task 5), KHÔNG hybrid,
            KHÔNG rerank — dùng làm baseline so sánh A/B
    """

    def __init__(self, config: str = "hybrid_rerank", top_k: int = 5):
        if config not in ("hybrid_rerank", "dense_only"):
            raise ValueError(f"Unknown config: {config}")
        self.config = config
        self.top_k = top_k

    def generate_with_citation(self, question: str) -> dict:
        if self.config == "hybrid_rerank":
            chunks = retrieve(question, top_k=self.top_k)
        else:  # dense_only
            chunks = semantic_search(question, top_k=self.top_k)

        return _generate_with_citation(question, context_chunks=chunks)


# =============================================================================
# DeepEval judge model — trỏ sang OpenRouter (không phải OpenAI trực tiếp)
# =============================================================================

def _build_judge_model():
    """
    DeepEval CHƯA hỗ trợ native OpenRouter (github.com/confident-ai/deepeval
    issue #2626, còn mở tính đến thời điểm viết code này). Cách chính thức
    khả dụng: dùng `LocalModel` (thiết kế cho LM Studio/vLLM — nhưng nhận
    base_url tùy ý, hoạt động với BẤT KỲ endpoint OpenAI-compatible nào,
    OpenRouter cũng tương thích wire-format OpenAI nên dùng được).
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Chưa set OPENROUTER_API_KEY trong .env (cùng key đã dùng ở Task 10)."
        )

    if EVAL_MODEL.endswith(":free"):
        print(
            f"  ⚠ CẢNH BÁO: EVAL_MODEL='{EVAL_MODEL}' là model FREE-TIER của "
            f"OpenRouter — giới hạn 50 request/ngày CHO CẢ TÀI KHOẢN. Với "
            f"nhiều câu hỏi x nhiều metric, rất dễ hết quota giữa chừng. "
            f"Cân nhắc đổi sang model trả phí rẻ (vd 'openai/gpt-4o-mini') "
            f"hoặc giảm MAX_QUESTIONS xuống số nhỏ."
        )

    from deepeval.models import LocalModel
    return LocalModel(model=EVAL_MODEL, base_url="https://openrouter.ai/api/v1", api_key=api_key)


# =============================================================================
# Option: DeepEval — chấm điểm TỪNG metric riêng lẻ (không dùng evaluate()
# batch) để sống sót khi bị rate-limit giữa chừng.
# =============================================================================

def evaluate_with_deepeval(rag_pipeline: RAGPipeline, golden_dataset: list[dict]) -> list[dict]:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    Returns:
        List[dict] — mỗi dict là 1 hàng kết quả:
        {question_id, question, answer, faithfulness, answer_relevance,
         context_recall, context_precision}
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    judge_model = _build_judge_model()

    metrics_by_key = {
        "faithfulness": FaithfulnessMetric(threshold=0.7, model=judge_model),
        "answer_relevance": AnswerRelevancyMetric(threshold=0.7, model=judge_model),
        "context_recall": ContextualRecallMetric(threshold=0.7, model=judge_model),
        "context_precision": ContextualPrecisionMetric(threshold=0.7, model=judge_model),
    }

    rows = []
    total = len(golden_dataset)

    for i, item in enumerate(golden_dataset, 1):
        question = item["question"]
        print(f"  [{rag_pipeline.config}] {i}/{total}: {question[:60]}...")

        row = {
            "config": rag_pipeline.config,
            "question_id": item["id"],
            "question": question,
        }

        # --- Bước 1: chạy RAG pipeline (retrieve + generate) ---
        try:
            result = rag_pipeline.generate_with_citation(question)
            answer = result["answer"]
            sources = result.get("sources", [])
        except Exception as e:
            print(f"    ⚠ Lỗi chạy pipeline: {e} — bỏ qua câu này")
            row.update({"answer": "", "faithfulness": None, "answer_relevance": None,
                        "context_recall": None, "context_precision": None})
            rows.append(row)
            continue

        row["answer"] = answer
        row["num_context_chunks"] = len(sources)

        retrieval_context = [c["content"] for c in sources] if sources else [""]
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=item["expected_answer"],
            retrieval_context=retrieval_context,
        )

        # --- Bước 2: chấm TỪNG metric riêng lẻ (không batch) ---
        # QUAN TRỌNG: nếu 1 metric bị rate-limit/lỗi, các metric/câu hỏi khác
        # ĐÃ chạy xong vẫn được giữ nguyên trong `rows` — không mất dữ liệu.
        for key, metric in metrics_by_key.items():
            try:
                metric.measure(test_case)
                row[key] = metric.score
            except Exception as e:
                print(f"    ⚠ Lỗi đo {key}: {e}")
                row[key] = None

        rows.append(row)

    return rows


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict], configs: list[str] | None = None) -> list[dict]:
    """
    So sánh A/B giữa các config retrieval.

    Args:
        golden_dataset: danh sách Q&A (đã áp dụng MAX_QUESTIONS nếu có)
        configs: list tên config, mặc định ["hybrid_rerank", "dense_only"]

    Returns:
        List[dict] gộp kết quả tất cả configs (thêm cột "config" để phân biệt)
    """
    if configs is None:
        configs = ["hybrid_rerank", "dense_only"]

    all_rows = []
    for config_name in configs:
        print(f"\n=== Config: {config_name} ===")
        pipeline = RAGPipeline(config=config_name)
        try:
            rows = evaluate_with_deepeval(pipeline, golden_dataset)
        except Exception as e:
            # Rate-limit/lỗi nghiêm trọng giữa chừng 1 config — vẫn giữ lại
            # kết quả của các config ĐÃ chạy xong trước đó thay vì crash mất
            # hết toàn bộ (all_rows tích luỹ từ các vòng lặp trước vẫn còn).
            print(f"  ✗ Config '{config_name}' dừng giữa chừng do lỗi: {e}")
            print(f"  → Vẫn xuất kết quả các config đã hoàn thành.")
            break
        all_rows.extend(rows)

    return all_rows


# =============================================================================
# Export Results
# =============================================================================

METRICS = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
}


def export_results(all_rows: list[dict]):
    """
    Export evaluation results ra results.md (+ CSV chi tiết để truy vết).

    Luôn ghi file dù `all_rows` rỗng/thiếu config — để không mất trắng dữ
    liệu đã thu thập được nếu chạy bị dừng giữa chừng (rate-limit).
    """
    if not all_rows:
        RESULTS_PATH.write_text(
            "# RAG Evaluation Results\n\n⚠ Không có kết quả nào (chạy bị lỗi "
            "ngay từ đầu hoặc golden_dataset rỗng).\n",
            encoding="utf-8",
        )
        print(f"⚠ Không có dữ liệu, đã ghi file rỗng: {RESULTS_PATH}")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(RAW_RESULTS_PATH, index=False, encoding="utf-8")
    print(f"✓ Đã lưu kết quả chi tiết: {RAW_RESULTS_PATH}")

    configs = sorted(df["config"].unique())
    summary = df.groupby("config")[METRICS].mean(numeric_only=True).round(3)
    n_failed = df[METRICS].isna().any(axis=1).sum()

    lines = ["# RAG Evaluation Results\n"]
    lines.append(
        f"Golden dataset: {df['question_id'].nunique()} câu hỏi · "
        f"Configs: {', '.join(configs)} · Framework: DeepEval "
        f"(judge model: `{EVAL_MODEL}`)\n"
    )
    if n_failed:
        lines.append(
            f"⚠ **{n_failed} dòng có ít nhất 1 metric bị lỗi/thiếu điểm** "
            f"(rate-limit hoặc lỗi API giữa chừng) — các dòng này bị loại "
            f"khỏi tính trung bình ở bảng dưới. Xem `eval_results_raw.csv` "
            f"để biết chi tiết câu hỏi nào bị ảnh hưởng.\n"
        )

    lines.append("## 1. Bảng điểm tổng hợp (điểm trung bình, thang 0–1)\n")
    header = "| Config | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |"
    sep = "|---" * (len(METRICS) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for config in configs:
        row = summary.loc[config]
        lines.append(f"| {config} | " + " | ".join(f"{row[m]:.3f}" for m in METRICS) + " |")
    lines.append("")

    if len(configs) == 2:
        c1, c2 = configs
        lines.append(f"## 2. So sánh A/B: `{c1}` vs `{c2}`\n")
        lines.append(f"| Metric | {c1} | {c2} | Chênh lệch ({c1} − {c2}) |")
        lines.append("|---|---|---|---|")
        for m in METRICS:
            v1, v2 = summary.loc[c1, m], summary.loc[c2, m]
            diff = v1 - v2
            arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "=")
            lines.append(f"| {METRIC_LABELS[m]} | {v1:.3f} | {v2:.3f} | {arrow} {diff:+.3f} |")
        lines.append("")
    elif len(configs) == 1:
        lines.append(
            f"## 2. So sánh A/B\n\n⚠ Chỉ có 1 config ({configs[0]}) hoàn thành "
            f"— config còn lại bị dừng giữa chừng (xem log lúc chạy). Chạy "
            f"lại `compare_configs()` cho config còn thiếu.\n"
        )

    lines.append("## 3. Worst Performers (Faithfulness thấp nhất mỗi config)\n")
    for config in configs:
        lines.append(f"### {config}\n")
        subset = df[(df["config"] == config) & df["faithfulness"].notna()]
        worst = subset.nsmallest(3, "faithfulness")
        lines.append("| Question ID | Faithfulness | Câu hỏi |")
        lines.append("|---|---|---|")
        for _, r in worst.iterrows():
            q = r["question"].replace("|", "/")
            lines.append(f"| {r['question_id']} | {r['faithfulness']:.2f} | {q} |")
        lines.append("")

    lines.append("## 4. Phân tích\n")
    lines.append(
        "_(Điền tay sau khi đọc bảng số liệu ở trên)_\n\n"
        "- Config nào Faithfulness cao hơn — có khớp kỳ vọng (hybrid+rerank "
        "nên cao hơn dense-only)?\n"
        "- Context Precision của dense-only thấp hơn rõ rệt không (thiếu "
        "bước lọc bằng cross-encoder rerank)?\n"
        "- Context Recall thấp ở câu nào — do corpus thiếu tài liệu hay do "
        "chunking/embedding chưa tối ưu?\n"
    )
    lines.append("## 5. Đề xuất cải tiến\n")
    lines.append(
        "_(Điền tay dựa trên mục 4)_\n\n"
        "- Bổ sung corpus cho chủ đề có Context Recall thấp\n"
        "- Điều chỉnh SCORE_THRESHOLD ở Task 9 nếu cần\n"
        "- Thử chunk_size khác ở Task 4 nếu Context Precision thấp\n"
    )

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Đã sinh {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases"
          + (f" (giới hạn MAX_QUESTIONS={MAX_QUESTIONS})" if MAX_QUESTIONS else ""))

    all_rows = []
    try:
        all_rows = compare_configs(golden_dataset)
    except KeyboardInterrupt:
        print("\n⚠ Bị ngắt (Ctrl+C) — vẫn xuất kết quả đã thu thập được.")
    finally:
        # LUÔN export dù bị lỗi/ngắt giữa chừng — không để mất trắng dữ liệu
        # đã tốn quota/chi phí để chạy.
        export_results(all_rows)