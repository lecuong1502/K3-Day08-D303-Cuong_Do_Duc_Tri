"""
Task 7 — Reranking Module.

Phương pháp CHÍNH được chọn: Cross-encoder reranker —
jina-reranker-v2-base-multilingual, gọi qua Jina AI Reranker API (cloud),
không chạy local — không cần GPU/tải model, chỉ cần API key.


=== CÁC PHƯƠNG PHÁP KHÁC (đã implement đầy đủ, dùng khi cần) ===
- MMR: tự implement bằng cosine similarity thuần Python/numpy, không cần model
  ngoài — dùng embedding đã có sẵn từ Task 5 (bge-m3 qua Ollama).
- RRF: tự implement, không cần API key, dùng để fuse BM25 (Task 6) +
  semantic search (Task 5) — sẽ dùng lại ở Task 9.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import math
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Tự tìm và load file .env ở thư mục gốc project (đi ngược lên từ vị trí file
# này, vì script có thể được chạy từ nhiều thư mục khác nhau, vd `cd src && python ...`
# hay chạy từ gốc project `python src/task7_reranking.py`).
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

JINA_API_URL = "https://api.jina.ai/v1/rerank"
JINA_MODEL = "jina-reranker-v2-base-multilingual"


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates bằng Jina Reranker API (jina-reranker-v2-base-multilingual).

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored (key 'score' = relevance_score
        do Jina trả về, khoảng [0,1], càng gần 1 càng liên quan) và sorted
        descending.
    """
    if top_k <= 0 or not candidates:
        return []

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise ValueError(
            f"JINA_API_KEY chưa được set. Lấy key tại https://jina.ai/reranker/ "
            f"rồi tạo file {_ENV_PATH} với nội dung:\n"
            f"  JINA_API_KEY=jina_xxxxxxxxxxxxxxxxxxxxx"
        )

    try:
        response = requests.post(
            JINA_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_MODEL,
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": min(top_k, len(candidates)),
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        raise RuntimeError(f"Jina API lỗi: {e} — chi tiết: {body}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Không gọi được Jina API (kiểm tra mạng): {e}")

    results = response.json()["results"]
    return [
        {
            **candidates[r["index"]],
            "score": r["relevance_score"],
        }
        for r in results
    ]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query (dùng embed_text() ở Task 4/5,
            model bge-m3 qua Ollama, để cùng không gian vector với candidates)
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
            LƯU Ý: mỗi candidate PHẢI có key 'embedding' — semantic_search() ở
            Task 5 hiện KHÔNG trả embedding trong kết quả (chỉ trả content/score/
            metadata để nhẹ), nên cần tự embed lại content hoặc sửa Task 5 để
            include=["embeddings", ...] khi query ChromaDB nếu muốn dùng MMR.
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR (giữ nguyên score gốc,
        không ghi đè bằng mmr_score — vì mmr_score không phải similarity thật,
        chỉ dùng để CHỌN thứ tự).
    """
    if top_k <= 0 or not candidates:
        return []

    missing = [i for i, c in enumerate(candidates) if "embedding" not in c]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(candidates)} candidates thiếu key 'embedding'. "
            "MMR cần embedding cho mọi candidate — xem docstring."
        )

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker,
            vd [semantic_search(...), lexical_search(...)])
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            content = item["content"]
            scores[content] = scores.get(content, 0.0) + 1.0 / (k + rank)
            items[content] = item

    ranked_contents = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    results = []
    for content in ranked_contents:
        result = items[content].copy()
        result["score"] = scores[content]
        results.append(result)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
    **kwargs,
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking
        **kwargs: tham số riêng cho từng method
            - mmr: query_embedding (bắt buộc), lambda_param (tùy chọn)
            - rrf: ranked_lists (bắt buộc, thay cho candidates)

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        query_embedding = kwargs.get("query_embedding")
        if query_embedding is None:
            raise ValueError("method='mmr' cần truyền query_embedding=... trong kwargs")
        return rerank_mmr(
            query_embedding, candidates, top_k,
            lambda_param=kwargs.get("lambda_param", 0.7),
        )
    elif method == "rrf":
        ranked_lists = kwargs.get("ranked_lists")
        if ranked_lists is None:
            raise ValueError("method='rrf' cần truyền ranked_lists=[...] trong kwargs")
        return rerank_rrf(ranked_lists, top_k, k=kwargs.get("k", 60))
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]

    print("=== Cross-encoder (Jina Reranker API) ===")
    try:
        results = rerank_cross_encoder("tuition fee payment", dummy_candidates, top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content']}")
    except (ValueError, RuntimeError) as e:
        print(f"⚠ {e}")

    print("\n=== RRF (giả lập 2 ranker cùng list, top_k=2) ===")
    results = rerank_rrf([dummy_candidates, list(reversed(dummy_candidates))], top_k=2)
    for r in results:
        print(f"[{r['score']:.5f}] {r['content']}")