# Quyết định 01 – Chiến lược đến 17:00 VN (2026-10-22)

Người quyết định: Opus 5.5. Lập lúc 10:40 VN.

## Căn cứ mới (đã xác minh từ trang cuộc thi qua `kaggle competitions pages list rmit-hackathon-2026 --content`)

- Danh sách model được phép (10 model, đều là mirror `aisuko/...`): distilbert-base-multilingual-cased, mmBERT-small, **mmBERT-base**, **xlm-roberta-base**, **xlm-roberta-large**, rembert, deberta-v3-base-prompt-injection-v2, wolf-defender-prompt-injection, granite-3.3-2b-instruct, Mistral-7B-Instruct-v0.3. "No other pretrained backbone is allowed."
- Phải xây lời giải trên **một** model trong danh sách và **ghi rõ model đã chọn ở đầu notebook**.
- Tổng cộng chỉ **6 lượt nộp** (CLI báo còn 6 hôm nay, đã dùng 0).
- Nộp bài hợp lệ đầu tiên là được trọn điểm Challenge 1, sau đó mới cạnh tranh trên leaderboard.
- Trang "rules" chỉ ghi: "Submit notebooks to get higher score." => ta nộp từ output của notebook `RMIT_SGS_Lock-In_C1` (`kaggle competitions submit -k ... -v N -f submission.csv`), không upload CSV rời.

## Quyết định

| # | Quyết định | Lý do |
|---|---|---|
| D1 | Thời gian thực (6h20) là ràng buộc, không phải GPU-giờ (30h). Chạy **2 phiên GPU song song**: kernel chính M = `RMIT_SGS_Lock-In_C1` (mọi bài nộp lấy từ đây) và kernel thử nghiệm E = `RMIT_SGS_LockInExp_C1` (đổi ở decisions/02) (private, không bao giờ dùng để nộp). | Tối đa 2 slot × 6h ≈ 12 GPU-giờ, dư quota. Nộp chỉ từ M để chắc chắn đúng tên notebook. |
| D2 | **M1 = xlm-roberta-base trước** (K=3, 3 epoch, LR 2e-5), xong thì nộp ngay làm bảo hiểm. | Lấy trọn điểm C1 sớm (~12:00) và có mốc public score. |
| D3 | **M2 = xlm-roberta-large** (K=3, 3 epoch, LR 1e-5, warmup 10%), là ứng viên chính để nộp cuối. | Encoder mạnh nhất trong danh sách, khả thi trên T4 x2 với head+tail 512 token. |
| D4 | **E1 = mmBERT-base** (K=3, 3 epoch, LR 3e-5, max 512 head+tail) chạy song song với M1. | Model hiện đại, đa ngôn ngữ, nhanh; nếu thắng large thì chạy lại trong M ở M3. Mirror chỉ có `pytorch_model.bin` (không có safetensors) => code tìm model phải chịu được cả 2 loại file. |
| D5 | K=3 cho mọi run. | So sánh OOF công bằng giữa các model; K=5 cho large gấp ~2 lần thời gian, không kịp. |
| D6 | Không ensemble khác kiến trúc, không pseudo-label, không chỉnh tay dữ liệu test. Chỉ trung bình các fold của **cùng một** model. | Luật yêu cầu "one of the approved models"; luật nền tảng 4b cấm gán nhãn tay test. |
| D7 | Notebook phải có **giới hạn thời gian trong code** (`TIME_BUDGET_MIN`): sau mỗi fold, nếu fold tiếp theo dự kiến vượt ngân sách thì dừng, vẫn ghi `submission.csv` từ các fold đã xong. Push luôn kèm `-t` (giây) lớn hơn ngân sách 15 phút. | CLI không có lệnh hủy run; phải tự chặn để không mất trắng output. |
| D8 | Bỏ ý tưởng granite/Mistral LoRA. | Không đủ thời gian để thử và debug trước 17:00. |

## Tiêu chí cổng nộp (OOF = macro F1 OOF sau khi tinh chỉnh trọng số lớp)

- **Gate A (nộp #1, base):** run thành công, `submission.csv` qua assert định dạng, OOF ≥ 0.85. Nếu OOF < 0.85 => có lỗi pipeline, không nộp, báo Opus.
- **Gate B (các lần nộp sau):** OOF mới ≥ OOF tốt nhất đã nộp + 0.003, run đủ cả K fold (không phải bản dừng sớm), assert định dạng đạt.
- **Kiểm tra sau nộp:** nếu public score mới < public tốt nhất − 0.005 thì không nộp thêm biến thể cùng hướng.
- Ngân sách lượt nộp: dự kiến dùng tối đa 4 (#1 base, #2 large, #3 tùy chọn), giữ ≥ 2 lượt dự phòng lỗi.
- Người nộp: thành viên khác của team. Agent không chạy `kaggle competitions submit`.

## Điều kiện quay lại hỏi Opus

- Smoke lỗi 2 lần liên tiếp với log trống.
- Thời gian 1 fold large (đo từ smoke hoặc log M2) dự kiến làm M2 kết thúc sau 15:30 VN.
- OOF large < OOF base, hoặc một fold có macro F1 < 0.6 (dấu hiệu sụp huấn luyện).
- Phiên E bị xếp hàng/không chạy song song được.
- mmBERT-base OOF > large OOF.
