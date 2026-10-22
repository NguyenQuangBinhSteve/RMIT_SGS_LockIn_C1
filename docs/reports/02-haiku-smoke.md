# Báo cáo 02 – Smoke M (xlm-roberta-large) và smoke E (mmBERT-base)

Người chạy: Haiku 5.5 (high). Runbook: `plans/03-chay-thuc-nghiem-haiku.md`. Kế hoạch: `plans/01-ke-hoach-gpu.md`.

## Kết quả tóm tắt

| Run | Kernel | Version | Trạng thái | Giờ push → RUNNING → xong (VN) | OOF macro F1 (raw / tuned) |
|---|---|---|---|---|---|
| m-smoke | `qucanhtrnnguyn/rmit-sgs-lock-in-c1` | v4 | ERROR, log trống | 10:45:05 → 10:45 → 10:46 | – |
| m-smoke | `qucanhtrnnguyn/rmit-sgs-lock-in-c1` | v5 (push lại, không đổi) | COMPLETE | 10:50:43 → 10:51 → 10:56 | 0.3959 / 0.4697 |
| e-smoke | `qucanhtrnnguyn/lockin-c1-exp` | v1 | COMPLETE | 10:45:18 → 10:45 → 10:52 | 0.6444 / 0.6586 |

Hai kernel chạy song song được (M và E đều RUNNING cùng lúc, 10:51–10:52 và 10:51–10:56).

## Lệnh đã chạy

Không sinh lại notebook. Dùng thư mục đã sinh sẵn:

```
$K kernels push -p $S/gen/large-main-smoke   --accelerator NvidiaTeslaT4 -t 1500   # M, v4 rồi v5
$K kernels push -p $S/gen/mmbert-exp-smoke  --accelerator NvidiaTeslaT4 -t 1500   # E, v1
```

`$S` là thư mục scratchpad, `$K` là `$S/venv/bin/kaggle` với `KAGGLE_API_TOKEN` từ `~/.kaggle/access_token`.

Bước 0 (chẩn đoán v3): `kernels status` trả về ERROR, `kernels output` và `kernels logs` đều trống (`[]`). File `reports/00-haiku-buoc0-log-v3.md` không tồn tại tại thời điểm chạy.

Kiểm tra trước khi push:
- `kernel-metadata.json` của cả 2 thư mục: `enable_internet: false`, `enable_gpu: true`, `model_sources` đúng 1 model (`aisuko/xlm-roberta-large/transformers/default/1` và `aisuko/mmbert-base/transformers/default/1`), `competition_sources: rmit-hackathon-2026`.
- Notebook JSON hợp lệ (6 cell).
- Cờ trong notebook: `SMOKE=True`, budget 140 (M) / 55 (E), `HEAD=384`, `TAIL=126`. Smoke dùng 800 dòng train (lấy mẫu), K=2, epochs=1, test 300 dòng.

Chưa push lần nào vào kernel đang RUNNING/QUEUED. Không chạy `kaggle competitions submit`, không git, không sửa code.

## Chi tiết run

### M – m-smoke (xlm-roberta-large)
- Trạng thái: v4 ERROR với log trống (10:46). Đợi 3 phút, `kernels logs` vẫn trống. `kernels pull -m` cho thấy metadata khớp bản đã push (chỉ khác `Transformers` viết hoa trong `model_sources`, đây là chuẩn hóa của Kaggle). Notebook pull về mở được (6 cell). Theo runbook push lại 1 lần không đổi: v5 chạy thành công.
- Lỗi v4 không có traceback. Không rõ nguyên nhân (nghi lỗi phía Kaggle, tạm thời).
- Trạng thái cuối v5: COMPLETE, `stopped_early: false`, folds 2/2.
- GPU: 2 × Tesla T4; torch 2.11.0+cu128; transformers 5.16.1.
- Model: `FacebookAI/xlm-roberta-large` (mirror `aisuko/xlm-roberta-large`).
- OOF macro F1: raw 0.3959 | tuned 0.4697 | weights [1.0, 1.0, 1.5].
- F1 từng fold: 0.3387, 0.4240. Phút train từng fold: 1.2, 1.3 (`train_min`). Phút từng fold (kể cả predict): 2.05, 1.82. Tổng 3.95 phút.
- F1 theo độ dài: 0–400: 0.5063 (442 dòng); 400–800: 0.3150 (179); 800+: 0.3446 (179).
- F1 theo ngôn ngữ: en 0.4249 (83); fr 0.4576 (189); it 0.4334 (169); pt 0.5040 (359).
- Confusion matrix OOF: [[314, 168, 17], [99, 145, 0], [25, 22, 10]].
- Phân bố nhãn test (300 dòng smoke): JAILBREAK 176 (58.7%), BENIGN 124 (41.3%), PROMPT_INJECTION 0 (0%). Smoke có 400 dòng train/fold và 1 epoch nên không đánh giá được phân bố này (kỳ vọng 5–9%).
- `submission.csv`: 300 dòng, cột `id,label` (đúng định dạng smoke; assert 6000 dòng chỉ chạy khi không smoke).
- Cảnh báo tokenizer "sequence longer than max length (796 > 512)" trước khi cắt head/tail: vô hại theo báo cáo Sonnet, đã cắt về 512 trước khi đưa vào model.
- Lỗi: không có traceback trong log v5.

### E – e-smoke (mmBERT-base)
- Trạng thái cuối: COMPLETE, `stopped_early: false`, folds 2/2. Không cần push lại.
- GPU: 2 × Tesla T4; torch 2.11.0+cu128; transformers 5.16.1.
- Model: `jhu-clsp/mmBERT-base` (mirror `aisuko/mmbert-base`), `ModernBertForSequenceClassification`. Load report chỉ cảnh báo `classifier.*` MISSING (khởi tạo mới) và `decoder.*` UNEXPECTED: bình thường khi đổi head, không phải lỗi.
- OOF macro F1: raw 0.6444 | tuned 0.6586 | weights [1.0, 0.5, 0.75].
- F1 từng fold: 0.6786, 0.6139. Phút train từng fold: 0.6, 0.6 (`train_min`); phút từng fold: 4.25 (fold 0, có khởi động CUDA/load), 0.87 (fold 1). Tổng 5.27 phút.
- Train loss: 1.939 (fold 0), 2.462 (fold 1). Không thấy NaN trong loss; chưa kiểm tra riêng fp16/gradient checkpointing trong log (rủi ro #1 trong báo cáo Sonnet chưa được xác nhận hay loại trừ bằng log).
- F1 theo độ dài: 0–400: 0.6926 (442); 400–800: 0.4322 (179); 800+: 0.4845 (179).
- F1 theo ngôn ngữ: en 0.4653 (83); fr 0.6940 (189); it 0.7750 (169); pt 0.6373 (359).
- Confusion matrix OOF: [[404, 81, 14], [69, 173, 2], [32, 2, 23]].
- Phân bố nhãn test (300 dòng): BENIGN 199 (66.3%), JAILBREAK 93 (31.0%), PROMPT_INJECTION 8 (2.7%).
- `submission.csv`: 300 dòng, cột `id,label`.
- Lỗi: không có traceback. Cảnh báo `MissingIDFieldWarning` của nbformat và `SyntaxWarning` của nbconvert: vô hại.

## GPU-giờ (`kaggle quota`)
- Trước: GPU 0.01h used / 29.99h remaining (trước khi push).
- Sau khi E và M (v5) xong: GPU 0.19h used / 29.81h remaining.
- Smoke M v4 (ERROR, không có log) có thể đã tốn thêm một ít. Quota chỉ tăng 0.18h cho cả 3 phiên (M v4 + M v5 + E v1), phù hợp với thời lượng thực tế. Ghi nhận: quota hiện không phân biệt phiên lỗi.
- Refresh quota: 2026-10-24T00:00:00.

## Ngoại suy thời gian bản full (theo công thức runbook: phút train 1 fold full ≈ 70 × phút train 1 fold smoke)

Dùng `train_min` (không gồm load/predict). Ước lượng thêm ~3 phút/fold cho load và predict full (6000 test + ~3100 validation). Đây là ước lượng, không phải số đo.

| Cấu hình | train_min smoke | train 1 fold full (70×) | Mỗi fold (ước lượng, +3 phút) | Ngân sách | Số fold chạy được theo `can_continue` |
|---|---|---|---|---|---|
| large, epochs=3 (M2 mặc định) | 1.25 (TB 1.2, 1.3) | ≈ 88 phút | ≈ 91 phút | 140 | 1 fold (91 + 91 > 140, dừng sau fold 1) |
| large, epochs=2 | 1.25 | ≈ 58 phút | ≈ 61 phút | 140 | 2 fold (fold 1 ≈ 61, 61+61 ≤ 140; fold 2 xong ≈ 122, 122+61 > 140, dừng) |
| large, epochs=2, ba fold đầy đủ | 1.25 | ≈ 58 × 3 | ≈ 183 phút tổng | cần ≥ ~185 | 3 fold |
| mmBERT-base, epochs=3 (E1, M1 base) | 0.6 | ≈ 42 phút | ≈ 45 phút | 55 | 1 fold (45 + 45 > 55, dừng sau fold 1) |
| mmBERT-base, epochs=3, ba fold đầy đủ | 0.6 | ≈ 42 × 3 | ≈ 135 phút tổng | cần ≥ ~135 | 3 fold |

Công thức `can_continue(elapsed, fold_mins, budget)` là `elapsed + mean(fold_mins) <= budget`, kiểm sau mỗi fold trừ fold cuối.

Hệ quả cho runbook (chỉ báo, không tự sửa):
1. Với `--budget 55` (M1 base, E1 mmBERT) sẽ chỉ chạy 1/3 fold, OOF chỉ phủ 1/3 train. Muốn đủ 3 fold thì budget cần ≥ ~135 phút, và `-t` cũng phải tăng (`-t 4200` = 70 phút, không đủ).
2. Với `--budget 140` (M2 large): kể cả `EPOCHS=2` vẫn không đủ 3 fold (~183 phút). Theo điều kiện dừng 2 của plan 01, cần Opus quyết: nâng budget và `-t` (ít nhất ~190 phút, tức ~11400 s, thay vì 9300 s), hoặc chạy K=2, hoặc bỏ large.
3. `-t 9300` (155 phút) không đủ cho 3 fold large với `EPOCHS=2` (~183 phút). Kernel sẽ bị kill giữa chừng nếu chạy đủ 3 fold.

## Lỗi
- M v3 (trước đó): ERROR, log trống. Không có báo cáo chẩn đoán `reports/00-haiku-buoc0-log-v3.md`.
- M v4 (10:45): ERROR, log trống, metadata khớp, notebook JSON hợp lệ. Không có traceback. Push lại không đổi (v5, 10:50): COMPLETE. Nghi lỗi tạm thời phía Kaggle, không phải code.
- E v1: không lỗi.
- Không có traceback nào từ code. Không có NaN, không có OOM.

## Đề xuất (cho Opus quyết)
1. Smoke cả hai đều chạy được end-to-end trên T4 x2 với cấu hình hiện tại. Có thể chuyển sang M1/E1 (base/mmBERT) nếu Opus chấp nhận rủi ro budget.
2. Quyết định trước khi chạy M1/E1: budget 55 chỉ cho 1 fold. Chọn một trong hai: nâng budget ≥ 135 và `-t` tương ứng, hoặc giữ 55 và chấp nhận OOF 1/3.
3. Quyết định trước khi chạy M2: large với `EPOCHS=2` cần ~183 phút. Nâng budget/`-t` (≥ ~190 phút) hay bỏ large (điều kiện dừng 2 của plan 01).
4. Smoke OOF không dùng để so sánh chất lượng (400 dòng train, 1 epoch). Số liệu đáng chú ý: mmBERT F1 0.66 tuned, xlm-roberta-large 0.47 tuned trong cùng điều kiện smoke. Chưa kết luận gì về full.
5. Lỗi log trống gặp 2 lần trên M (v3, v4), v5 chạy được. Nếu còn gặp lần nữa, theo runbook dừng và báo Opus.
