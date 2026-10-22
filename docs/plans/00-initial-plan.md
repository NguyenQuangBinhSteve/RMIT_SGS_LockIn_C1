# Kế hoạch RMIT Hackathon 2026 – Challenge 1 (Team Lock In)

Phân loại prompt đối kháng đa ngôn ngữ 3 lớp: BENIGN / JAILBREAK / PROMPT_INJECTION. Metric: macro F1.

## 1. Thông tin cố định

- Competition: `rmit-hackathon-2026` (deadline 2026-10-22 11:00 UTC). Dừng cứng lúc 10:15 UTC.
- Kaggle username: `qucanhtrnnguyn`. Tên team: Lock In (campus SGS).
- Kernel: `qucanhtrnnguyn/rmit-sgs-lock-in-c1`, title `RMIT_SGS_Lock-In_C1` (đúng quy ước đặt tên của ban tổ chức), private, GPU, internet tắt.
- Model được phép (mirror của ban tổ chức): `aisuko/xlm-roberta-base`, `aisuko/xlm-roberta-large`. Dùng trong kernel qua `model_sources`: `aisuko/xlm-roberta-large/transformers/default/1`.
- Bắt buộc fine-tune model; không pseudo-label, không ensemble nhiều kiến trúc (quy định còn mơ hồ).
- Dữ liệu: train 14000 dòng (BENIGN 8680, JAILBREAK 4340, PROMPT_INJECTION 980; ngôn ngữ pt 6300, fr 3500, it 2800, en 1400), test 6000 dòng. Cột train: `id,text,language,label`; test: `id,text,language`.
- Chưa có bài nộp nào trước đó.

## 2. Quyết định của người dùng

- Được phép push kernel lên Kaggle bằng tài khoản `qucanhtrnnguyn`.
- Chỉ nộp khi điểm tốt hơn đợt trước. Public score chỉ biết sau khi nộp, nên dùng OOF macro F1 làm cổng trước khi nộp; sau khi nộp mà public score không cao hơn thì không nộp thêm biến thể tương tự.
- Việc chạy kiểm tra bài nộp do thành viên khác của team làm. Agent/Claude không tự chạy `kaggle competitions submit` khi chưa qua cổng và chưa được người dùng đồng ý.
- Không tự chạy `git commit` / `git push`; chỉ đưa commit message cho người dùng.

## 3. Quan sát dữ liệu (ảnh hưởng thiết kế)

- Template lặp: prefix 25 ký tự gom thành 1025 nhóm, 408 dòng có nhãn lẫn trong nhóm (9.6%); 30% test trùng prefix với train. => dùng StratifiedGroupKFold theo prefix 60 ký tự đã chuẩn hóa để OOF không bị thổi phồng.
- Nhãn phụ thuộc độ dài: PROMPT_INJECTION tập trung ở văn bản ngắn (0–400 ký tự), JAILBREAK chiếm ưu thế ở 200–700 ký tự, BENIGN/JAILBREAK chia ~50/50 ở văn bản >500 ký tự. Độ dài là tín hiệu yếu ở vùng dài.
- Nhiều BENIGN chứa câu "ignore instructions" (vd "veuillez ignorer", "si prega di ignorare") giống JAILBREAK về từ vựng => cần hiểu ngữ nghĩa.
- Trần ước lượng ≤ ~0.96 do nhiễu nhãn.
- Median độ dài 294 ký tự, max 18550 => cắt head+tail (384 + 126 token + token đặc biệt = 512).

## 4. Lộ trình

### Bước 1 – Chuẩn bị (~20 phút)
- Cài `kaggle` CLI trong venv, xác thực bằng `~/.kaggle/access_token`.
- Notebook `rmit-hackathon-2026/RMIT_SGS_Lock-In_C1.ipynb`: model large, head+tail 512, `group_by_length`, output `/tmp/out`, StratifiedGroupKFold theo prefix 60 ký tự, cờ `SMOKE`, lưu `oof.npy` và `test_probs.npy`, log confusion matrix theo độ dài và ngôn ngữ, test = trung bình xác suất các fold (không refit riêng).
- `kernel-metadata.json` với GPU, tắt internet, gắn competition + model source.

### Bước 2 – Smoke test (~15 phút)
- Push kernel với `SMOKE=True` (800 dòng, 2 fold, 1 epoch, 300 dòng test), đo thời gian mỗi epoch.
- Lỗi thì sửa rồi push lại. Không nộp.

### Bước 3 – Run 1: xlm-roberta-base, group fold (~45 phút)
- Lấy OOF macro F1 làm mốc.
- **Gate A:** OOF ≥ 0.94 thì nộp bản này làm bảo hiểm và lấy public score làm mốc. Thấp hơn thì dừng, phân tích dữ liệu trước khi tốn thêm GPU.

### Bước 4 – Run 2: xlm-roberta-large, 3 fold (~2–3.5 giờ)
- **Gate B:** chỉ nộp nếu OOF cao hơn Run 1; nếu public score sau khi nộp không cao hơn mốc Run 1 thì không nộp biến thể tương tự.
- OOF ≥ 0.955: chốt large. OOF < 0.94: sang bước 5.

### Bước 5 – Phân tích lỗi và khám phá (~4 giờ, chỉ khi Gate B không đạt)
- Phân tích confusion matrix theo độ dài (0-400 / 400-800 / 800+) và ngôn ngữ để biết lỗi do ngữ nghĩa hay do phân bố.
- Tùy chọn: granite-3.3-2b-instruct + LoRA, chỉ khi xác nhận nằm trong danh sách model được phép.

### Bước 6 – Chốt (trước 10:15 UTC)
- Chọn bài cuối theo OOF và public score.
- Chừa 2–3 lượt nộp dự phòng cho lỗi định dạng.

## 5. Trạng thái hiện tại (2026-10-22, ~03:50 UTC)

- Kaggle CLI 2.2.4 đã cài tại `/tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad/venv/bin/kaggle`; xác thực bằng `export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)`. Đã xác nhận `userHasEntered=True`.
- Notebook đã được sinh lại bằng `scratchpad/mk_nb.py` (xlm-roberta-large, K=3, EPOCHS=3, LR 1e-5 cho large / 2e-5 cho base, HEAD=384, TAIL=126), `SMOKE=False`. Chưa chạy thử trên Kaggle ở dạng đầy đủ.
- Bản smoke (`scratchpad/push/`) đã push thành version 1 với `--accelerator NvidiaTeslaT4` và kết thúc ở trạng thái `ERROR`. Log tải về (`scratchpad/out1/rmit-sgs-lock-in-c1.log`) chỉ là `[]`, chưa biết nguyên nhân lỗi.
- Repo team `LockIn_RMIT_HACKATHON_2026` đã clone, mới chỉ có README.
