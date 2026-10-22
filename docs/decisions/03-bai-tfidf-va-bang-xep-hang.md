# Quyết định 03 – Bài TF-IDF 0.918 và bảng xếp hạng (11:15 VN)

## Sự kiện (đã kiểm bằng CLI)
- `submissions`: 1 bài trên tài khoản qucanhtrnnguyn: `baseline_tfidf_3class.csv`, 04:05 UTC, public 0.91819.
- `submission-limits`: đã dùng 1, **còn 5**.
- Top public: 0.984, phần lớn top 10 nằm ở 0.972–0.982.

## Rủi ro tuân thủ
1. ~~Người bạn có phải thành viên team không?~~ **Đã xác nhận (người dùng, ~11:20 VN): người bạn LÀ thành viên team Lock In.** Rủi ro theo luật nền 5a/5d/6a đã được loại bỏ. Các mục 2, 3 và mọi quyết định khác giữ nguyên.
2. **Bài TF-IDF không dùng model nào trong danh sách**, trong khi luật ghi "fine-tune one of the provided models". Bài này **không được làm bài cuối**. Kaggle tự chọn theo public, nên nếu mọi bài của ta < 0.918 thì Kaggle sẽ chọn đúng bài TF-IDF. => **Sửa Q3 (decisions/02): phải chọn tay bài cuối, chỉ trong các bài transformer từ kernel M.**
3. Bài TF-IDF có thể không được tính là "bài hợp lệ đầu tiên" cho điểm C1. Bài transformer đầu tiên vẫn cần nộp sớm.

## Q1 (sửa)
CLI chấp nhận CSV rời. Cách nộp ưu tiên vẫn là `-k qucanhtrnnguyn/rmit-sgs-lock-in-c1 -v N -f submission.csv`. Nếu CLI từ chối `-k` (CLI ghi `-k` dành cho code competition) thì tải `submission.csv` từ output kernel M version N rồi nộp `-f`, với `-m "RMIT_SGS_Lock-In_C1 vN, <model>, OOF x.xxxx"` để truy vết được về notebook.

## a. Cổng và mục tiêu (thay Gate A/B ở decisions/01)
- Trần ≤ 0.96 trong plan 00 là sai. Mục tiêu mới: **public ≥ 0.97**, tương ứng OOF group-fold khoảng ≥ 0.95. OOF group-fold thấp hơn public, vì 30% test có prefix trùng train mà group-fold cố ý loại.
- **Gate A (base):** OOF ≥ 0.92 và định dạng đạt => nộp (bài transformer đầu tiên). OOF < 0.92 thì không nộp, báo Opus.
- **Gate B:** giữ +0.003 so với OOF tốt nhất đã nộp. Thêm: public dự kiến phải > 0.918.
- Ngân sách 5 lượt: #1 base, #2 large, #3 dự phòng cho một cải tiến, 2 lượt cuối giữ cho lỗi định dạng.

## b. Hướng đột phá hợp luật
| Hướng | Quyết định | Ghi chú |
|---|---|---|
| Kết hợp TF-IDF/LR với transformer (stack, cộng xác suất, đặc trưng) | **KHÔNG** | Thành mô hình thứ hai, không thuộc danh sách. Trái "build your solution on one of the approved models" và D6. Rủi ro bị loại lớn hơn lợi ích. |
| Tra bảng prefix/template của train để gán nhãn test | **KHÔNG** | Là luật tay/kNN ngoài model, cùng rủi ro như trên. |
| Thêm thẻ ngôn ngữ vào đầu văn bản | Không làm ở M2 | Hiệu quả chưa chứng minh, không đủ thời gian đánh giá A/B. |
| Tinh chỉnh trọng số lớp/ngưỡng trên OOF | Đã có | Hợp luật (hậu xử lý xác suất của chính model). |
| Nhiều epoch hơn/LR | Theo t_base (plans/04) | Large 3 epoch nếu đủ thời gian. Template lặp nhiều nên model cần đủ epoch để học thuộc. |
| head+tail 512 | Giữ | – |
| K=4 song song (plans/04) | Giữ | Mỗi fold train trên nhiều dữ liệu hơn, giúp học template trùng test. |

Lý giải: TF-IDF đạt 0.918 cho thấy tín hiệu nằm nhiều ở từ vựng và template. Fine-tune large đủ epoch sẽ học được cả tín hiệu này, nên không cần ghép TF-IDF.

## c. Thứ tự run: không đổi
M1 base → Gate A mới → M2 large (K=4 song song nếu probe đạt). E1/E2 như cũ.

## Việc giao thêm
- **Sonnet S15:** `tools/tfidf_calib.py`, chạy cục bộ trên CPU, **chỉ để hiệu chuẩn, không bao giờ nộp**. Code do team tự viết, không dùng code của người bạn. Nội dung: TF-IDF char 2–5 + word 1–2, LogisticRegression `class_weight='balanced'`. In OOF macro F1 với (1) cùng StratifiedGroupKFold prefix-60, K=3, seed 42 như notebook, (2) StratifiedKFold thường. Chênh lệch (2) − (1) là mức group-fold làm OOF thấp hơn. 0.918 − OOF(1) là độ lệch ước lượng từ OOF sang public (thô, vì cấu hình của người bạn khác).
  - Dữ liệu: `kaggle competitions download rmit-hackathon-2026 -p $R/data`. **Người dùng nên thêm `data/` vào `.gitignore`**, không commit dữ liệu cuộc thi.
- **Haiku:** chạy S15, rồi viết `docs/reports/NN-tfidf-calib.md` gồm 2 số OOF và độ lệch ước lượng. Khi M1 xong, thêm vào báo cáo M1: F1 OOF tách theo hàng có prefix xuất hiện trong test và hàng không có, cùng confusion matrix.
- Opus dùng độ lệch này để chuyển OOF của M1/M2 sang public dự kiến trước khi duyệt cổng.
