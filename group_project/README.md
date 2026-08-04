# Bài Tập Nhóm — University Services RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:
- ✅ Yêu cầu 1: RAG Chatbot (Streamlit)
- ✅ Yêu cầu 2: RAG Evaluation Pipeline (DeepEval)

---

## Kiến Trúc Hệ Thống

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  PDF chính  │     │  Bài viết/  │     │   Convert   │     │   Chunk +    │
│    sách     ├────▶│  thông báo  ├────▶│  Markdown   ├────▶│ Embed + Index│
│  (crawl)    │     │  (crawl)    │     │             │     │  (ChromaDB)  │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬───────┘
                                                                     │
                            ┌────────────────────────────────────────┘
                            ▼
                  ┌───────────────────┐
                  │   Câu hỏi (Query)  │
                  └─────────┬─────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     ┌─────────────────┐         ┌─────────────────┐
     │  Semantic Search │         │  Lexical Search  │
     │    (ChromaDB)     │         │      (BM25)       │
     └────────┬─────────┘         └────────┬─────────┘
              │                             │
              └──────────────┬──────────────┘
                              ▼
                     ┌────────────────┐
                     │   Merge (RRF)   │
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │     Rerank      │
                     │ (Cross-encoder) │
                     └────────┬───────┘
                              │
              điểm thấp?      ▼
        ┌─────────────  Đủ liên quan? ─────────────┐
        │ (không)                        (có) │
        ▼                                      │
┌───────────────┐                              │
│   PageIndex     │                              │
│   (fallback)     │                              │
└───────┬───────┘                              │
        └───────────────────┬──────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  Context chunks  │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │    Reorder +     │
                    │  Format context  │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │   LLM (GPT-4o)   │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  Câu trả lời +   │
                    │    Citation      │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │   Streamlit UI   │
                    └─────────────────┘
```

### Stack chi tiết theo từng lớp

| Lớp | Công nghệ | Ghi chú |
|---|---|---|
| Thu thập dữ liệu | `requests`, Crawl4AI, MarkItDown | 10 PDF chính sách + 11 bài viết/trang HTML RMIT Vietnam |
| Chunking | `langchain-text-splitters` (Recursive, 500/50) | Đồng bộ giữa BM25 (Task 6) và semantic index (Task 4) |
| Embedding | `bge-m3:567m` qua Ollama local | Multilingual, hỗ trợ tốt cross-lingual VI↔EN |
| Vector store | ChromaDB (persistent, local) | Cosine similarity |
| Lexical search | BM25 (`rank-bm25`) | Cùng corpus đã clean với semantic search |
| Merge | Reciprocal Rank Fusion (RRF, k=60) | Tự implement |
| Rerank | Cross-encoder — Jina Reranker API | `jina-reranker-v2-base-multilingual` |
| Fallback | PageIndex (vectorless, tree-based retrieval) | Trigger khi cosine gốc < threshold |
| Generation | GPT-4o qua OpenRouter API | temperature=0.2, top_p=0.9 |
| UI | Streamlit | Conversation memory (giới hạn: chỉ ảnh hưởng generation, không ảnh hưởng retrieval) |
| Evaluation | DeepEval | Judge model: gpt-4o-mini (tách khỏi generation model, tránh self-preference bias) |

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Lê Kiên Cường | 2A202601427 | Task 1-10 (data pipeline + retrieval + generation), Streamlit chatbot, DeepEval evaluation | ✅ Hoàn thành |
| Xuân Thế Độ | 2A202601847 | Task 3 + Task 7 (chuyển file markdown + reranker) | ✅ Hoàn thành |
| Nguyễn Công Trí | 2A202601715 | Task 2 + Task 6 (crawl news data + BM25) | ✅ Hoàn thành |
| Trần Công Đức | 2A202601423 | Task 4 + Task 8 (chunking, indexing + PageIndex) | ✅ Hoàn thành |

---

## Hướng Dẫn Chạy

### 1. Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
pip install streamlit deepeval python-dotenv chromadb rank-bm25 langchain-text-splitters
```

### 2. Cấu hình `.env` (đặt ở gốc project)

```bash
cp .env.example .env
```

Điền các key sau vào `.env`:
```
JINA_API_KEY=jina_xxxxxxxxxxxxxxxxxxxxx
PAGEINDEX_API_KEY=pi_xxxxxxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxx
GENERATION_MODEL=openai/gpt-4o
EVAL_MODEL=openai/gpt-4o-mini
```

### 3. Chuẩn bị Ollama (embedding model, chạy local)

```bash
docker exec -it ollama ollama pull bge-m3:567m
```

### 4. Chạy pipeline dữ liệu (chỉ cần chạy 1 lần, hoặc khi corpus thay đổi)

```bash
cd src
python task1_collect_policies.py
python task2_crawl_news.py
python task3_convert_markdown.py
python task4_chunking_indexing.py
cd ..
```

### 5. Chạy chatbot (Yêu cầu 1)

```bash
streamlit run app.py
```

### 6. Chạy evaluation (Yêu cầu 2)

```bash
python group_project/evaluation/eval_pipeline.py
```

Kết quả: `group_project/evaluation/results.md` (bảng điểm + A/B comparison) và
`group_project/evaluation/eval_results_raw.csv` (chi tiết từng câu hỏi).

> ⚠️ Nếu dùng model OpenRouter dạng `:free`, giảm `MAX_QUESTIONS` trong
> `eval_pipeline.py` xuống số nhỏ (vd 5) để tránh vượt quota 50 request/ngày.

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.