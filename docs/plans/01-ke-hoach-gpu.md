# Kế hoạch 01 – Phân bổ GPU đến 17:00 VN (10:00 UTC), 2026-10-22

Xem quyết định tại `decisions/01-quyet-dinh.md`. Giờ dưới đây là giờ VN (UTC+7).

## Giả định

- Quota: 29.99h/30h GPU. Ràng buộc thật là thời gian thực (~6h20), không phải quota.
- Kaggle cho chạy song song **tối đa 2 phiên GPU** (chưa xác minh trên tài khoản này; nếu phiên thứ 2 bị xếp hàng thì bỏ slot E).
- Máy: `--accelerator NvidiaTeslaT4` (T4 x2). Thời lượng tối đa 1 phiên 12h, không ảnh hưởng.
- Ước lượng thô (sẽ thay bằng số đo từ smoke): base K=3×3 epoch ≈ 40–50 phút; mmBERT-base ≈ 40–50 phút; large K=3×3 epoch ≈ 2–2.5h.
- Ngoại suy từ smoke: smoke train 400 dòng × 1 epoch/fold; full train ~9333 dòng × 3 epoch/fold => **thời gian train 1 fold full ≈ 70 × thời gian train 1 fold smoke**, cộng ~2 phút predict test mỗi fold.

## Lịch (ĐÃ THAY bằng plans/01b-lich-sau-smoke.md)

| Giờ VN | Slot M: `RMIT_SGS_Lock-In_C1` (nộp từ đây) | Slot E: `RMIT_SGS_LockInExp_C1` (không nộp; xem decisions/02) | GPU-giờ ước tính | Người làm |
|---|---|---|---|---|
| 10:40–11:05 | Sonnet sửa generator (S1–S5); Haiku tải lại output v3 để tìm lỗi (không tốn GPU) | – | 0 | Sonnet, Haiku |
| 11:05–11:20 | Smoke large (`SMOKE=True`) | Smoke mmBERT-base | 0.3 | Haiku |
| 11:20–12:10 | **M1** base full, `TIME_BUDGET_MIN=55`, push `-t 4200` | **E1** mmBERT-base full, budget 55, `-t 4200` | 1.7 | Haiku |
| ~12:10 | **Gate A** → thành viên team nộp #1 (base) | – | – | Opus duyệt, teammate nộp |
| 12:15–14:45 | **M2** large full, budget 140, `-t 9300` | E2 (chỉ khi E1 OOF ≥ base OOF + 0.01): mmBERT-base HEAD=768/TAIL=254 (1024 token), budget 90. Ngược lại để trống. | 2.5 + ≤1.5 | Haiku |
| ~14:45 | **Gate B** → nộp #2 (large) nếu đạt | – | – | Opus, teammate |
| 14:50–16:25 | **M3** (tùy chọn): chạy lại cấu hình thắng ở E trong M nếu OOF E > OOF M2 + 0.003 và chạy được ≤ 90 phút. Không có thì để trống. | – | ≤1.6 | Haiku |
| 16:25 | **Dừng push mọi run mới.** | Dừng | – | – |
| 16:25–17:00 | Gate cho M3 → nộp #3 nếu đạt. Chọn bài cuối (OOF + public), ghi báo cáo chốt. | – | – | Opus, teammate |

Tổng dự kiến ≈ 6–8 GPU-giờ, dự phòng lỗi/chạy lại ≈ 2h => < 10h, dư nhiều so với 30h.

## Điều kiện dừng / đổi hướng (mục 2, 3, 5 đã thay bằng plans/01b)

1. **Smoke M lỗi log trống 2 lần** → dừng, báo Opus (nghi lỗi môi trường/metadata, không phải code).
2. **Ngoại suy từ smoke: large full > 140 phút** → M2 dùng `EPOCHS=2` (≈ 2/3 thời gian). Nếu vẫn > 140 phút → bỏ large, nộp bản tốt nhất giữa base và mmBERT (chạy lại mmBERT trong M nếu nó thắng).
3. **M2 không bắt đầu được trước 12:45** → M2 dùng `EPOCHS=2`.
4. **Fold 0 của M2 (xem `kaggle kernels logs`) có macro F1 < 0.6** → large bị sụp; báo Opus, cân nhắc LR 8e-6 hoặc bỏ large.
5. **Run dừng sớm do ngân sách** (in `STOP EARLY`) → bản đó không được nộp theo Gate B; có thể dùng làm dữ liệu so sánh.
6. **Sau 16:25** không push run mới, kể cả khi còn GPU.
7. **Gate A trượt (OOF < 0.85)** → không nộp, báo Opus phân tích confusion matrix trước khi chạy large.

## Checklist tuân thủ luật (kiểm trước mỗi lần nộp)

| # | Mục | Trạng thái |
|---|---|---|
| 1 | Backbone thuộc danh sách 10 model được phép, nạp từ mirror `aisuko/...` qua `model_sources` | Đã xác minh từ trang cuộc thi |
| 2 | Chỉ 1 model cho mỗi lời giải; chỉ trung bình fold cùng model; không ensemble khác kiến trúc | Áp dụng (D6) |
| 3 | Model đã chọn ghi rõ ở dòng đầu notebook (vd "Model: FacebookAI/xlm-roberta-large (mirror aisuko/xlm-roberta-large)") | Cần Sonnet sửa (S3) |
| 4 | Có fine-tune (không zero-shot) | Có |
| 5 | Tên notebook nộp: `RMIT_SGS_Lock-In_C1` (đúng mẫu `<Uni>_<Campus>_<Team>_C1`, campus SGS) | Đã đặt |
| 6 | Internet tắt (`enable_internet: false`) | Đã đặt |
| 7 | Nộp từ output của notebook M (`-k qucanhtrnnguyn/rmit-sgs-lock-in-c1 -v N -f submission.csv`) | Áp dụng |
| 8 | `submission.csv`: header `id,label`, đúng 6000 dòng, id khớp test, nhãn ∈ {BENIGN, JAILBREAK, PROMPT_INJECTION} | Assert trong notebook |
| 9 | Không gán nhãn tay / không dùng nhãn test; không pseudo-label | Áp dụng |
| 10 | Tổng ≤ 6 lượt nộp; dự kiến ≤ 4 | Theo dõi bằng `kaggle competitions submission-limits rmit-hackathon-2026` |
| 11 | Không chia sẻ code riêng ngoài team; kernel để private | Đã đặt |
| 12 | Thư viện mã nguồn mở (transformers, sklearn, torch) | Đạt |

## Câu hỏi cần người dùng xác nhận

1. "Submit notebooks to get higher score": ta hiểu là nộp từ output notebook (`-k`), không upload CSV rời. Đúng không?
2. Ban tổ chức có kiểm tra tên của **mọi** notebook trong tài khoản không? Kernel thử nghiệm `lockin-c1-exp` (private, không nộp) có chấp nhận được không? Nếu không, bỏ slot E và chạy tuần tự trong M.
3. Kaggle sẽ tự chọn bài cuối theo public score nếu không chọn tay. Người dùng có muốn chọn tay bài cuối (theo OOF) không?
