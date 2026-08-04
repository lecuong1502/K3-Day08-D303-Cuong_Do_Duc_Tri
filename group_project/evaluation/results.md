# RAG Evaluation Results

Golden dataset: 15 câu hỏi · Configs: dense_only, hybrid_rerank · Framework: DeepEval (judge model: `openai/gpt-4o-mini`)

## 1. Bảng điểm tổng hợp (điểm trung bình, thang 0–1)

| Config | Faithfulness | Answer Relevance | Context Recall | Context Precision |
|---|---|---|---|---|
| dense_only | 0.935 | 0.878 | 1.000 | 0.872 |
| hybrid_rerank | 0.961 | 0.939 | 0.867 | 0.872 |

## 2. So sánh A/B: `dense_only` vs `hybrid_rerank`

| Metric | dense_only | hybrid_rerank | Chênh lệch (dense_only − hybrid_rerank) |
|---|---|---|---|
| Faithfulness | 0.935 | 0.961 | ▼ -0.026 |
| Answer Relevance | 0.878 | 0.939 | ▼ -0.061 |
| Context Recall | 1.000 | 0.867 | ▲ +0.133 |
| Context Precision | 0.872 | 0.872 | = +0.000 |

## 3. Worst Performers (Faithfulness thấp nhất mỗi config)

### dense_only

| Question ID | Faithfulness | Câu hỏi |
|---|---|---|
| q04 | 0.67 | What is included in the Compulsory Non-Academic Fees for international students? |
| q11 | 0.75 | How can new students get their RMIT student ID card? |
| q07 | 0.78 | What is the process to register for courses on myRMIT? |

### hybrid_rerank

| Question ID | Faithfulness | Câu hỏi |
|---|---|---|
| q10 | 0.67 | What support services are available during RMIT Vietnam's Orientation Week? |
| q08 | 0.75 | Can students change or cancel their course enrolment after the deadline? |
| q01 | 1.00 | How are tuition fee payments made at RMIT Vietnam? |

## 4. Phân tích

- **So sánh Faithfulness và Answer Relevance:** Cấu hình `hybrid_rerank` đạt điểm Faithfulness (0.961) và Answer Relevance (0.939) cao hơn hẳn so với `dense_only` (tương ứng 0.935 và 0.878). Điều này hoàn toàn khớp với kỳ vọng: sự kết hợp giữa tìm kiếm từ khóa (BM25) và ngữ nghĩa (Vector), cộng thêm bước chấm điểm lại (Cross-encoder Reranker) đã giúp đẩy các đoạn văn bản (chunk) chứa thông tin chính xác và sát với câu hỏi nhất lên đầu. Nhờ đó, LLM có bối cảnh tốt hơn để trả lời đúng trọng tâm và giảm thiểu hallucination.
- **Về Context Precision:** Khá bất ngờ là Context Precision của cả hai cấu hình đều ngang nhau ở mức **0.872**. Việc thêm bước rerank không làm tăng hoặc giảm tỷ lệ tài liệu nhiễu trong tập top-K được đưa vào LLM. Tuy nhiên, cách reranker *sắp xếp lại thứ tự* (đưa chunk tốt nhất lên số 1, số 2) đã đủ để cải thiện chất lượng câu trả lời cuối (Answer Relevance tăng).
- **Về Context Recall:** Cấu hình `dense_only` đạt mức Recall tuyệt đối (1.000), trong khi `hybrid_rerank` lại giảm đáng kể xuống **0.867**. Nguyên nhân chính không hẳn do corpus thiếu tài liệu (vì `dense_only` vẫn tìm thấy), mà là do **bước Reranker hoặc thuật toán Hybrid đã "đánh giá thấp" và đẩy một số chunk hữu ích ra khỏi danh sách kết quả cuối cùng**. Điều này thể hiện rõ ở các câu bị điểm Faithfulness thấp của `hybrid_rerank` như **q10** (Orientation Week) và **q08** (cancel enrolment) — có thể LLM không nhận đủ thông tin nền tảng do chunk bị loại bỏ ở bước lọc.

## 5. Đề xuất cải tiến

- **Bổ sung và làm rõ Corpus cho các "Worst Performers":** Cần rà soát lại dữ liệu nguồn của các chủ đề bị điểm thấp. Cụ thể là các thông tin về *"Compulsory Non-Academic Fees"* (q04), *"Orientation Week support services"* (q10), và chính sách *"change/cancel course enrolment"* (q08). Đảm bảo các tài liệu này có chứa từ khóa rõ ràng để cả Dense và Lexical search đều có thể dễ dàng bắt được.
- **Điều chỉnh tham số Truy xuất và Rerank (Task 9):** 
  - Vì Context Recall của `hybrid_rerank` bị sụt giảm, cần xem xét **tăng số lượng `top_k` lấy từ bước hybrid retrieval** trước khi đưa vào reranker (ví dụ: lấy top 20 thay vì top 10), sau đó mới dùng reranker để cắt xuống top 5.
  - Thử nghiệm **hạ nhẹ `SCORE_THRESHOLD`** của mô hình reranker để tránh việc thuật toán chấm điểm quá khắt khe và vô tình loại bỏ các chunk chứa ngữ cảnh nền tảng.
- **Tối ưu hóa chiến lược Chunking (Task 4):** Phân tích các câu hỏi điểm thấp (như q07, q08, q11), ta thấy đa số là các câu hỏi về **quy trình (process) hoặc hướng dẫn (how-to)**. Việc Context Precision bị kẹt ở mức 0.872 có thể là do văn bản quy trình bị cắt đứt làm đôi ở bước chunking. Đề xuất **tăng `chunk_overlap`** hoặc thử một `chunk_size` lớn hơn để đảm bảo toàn bộ các bước của một quy trình (ví dụ: các bước đăng ký thẻ ID) nằm trọn vẹn trong một chunk.
