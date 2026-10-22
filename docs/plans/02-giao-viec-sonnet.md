# Kế hoạch 02 – Giao việc cho Sonnet 5.5 (high): sửa generator notebook

File cần sửa: `/tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad/mk_nb.py` (sinh notebook). Sau khi sửa, sinh lại `/home/quocanh/quoc_anh/rmit-hackathon-2026/RMIT_SGS_Lock-In_C1.ipynb` (cấu hình large, full).
Hạn: xong trước 11:05 VN. Giữ diff nhỏ nhất, không thêm thư viện.

## Ràng buộc

- Quy tắc comment: chỉ 1 dòng comment tiếng Việt ngay trước hàm/class mô tả chức năng; không comment mồ côi; sửa code có comment thì sửa comment đó, không thêm comment mới.
- Không ghi FR-xx/NFR-xx/SRS. Không git commit/push. Không chạy `kaggle kernels push` hay `kaggle competitions submit`.
- Không đổi thuật toán cốt lõi (StratifiedGroupKFold theo prefix 60 ký tự, cross-entropy có trọng số lớp, tinh chỉnh trọng số lớp trên OOF, test = trung bình các fold).

## Việc cần làm

**S1 – Tham số dòng lệnh.** `python mk_nb.py OUT_DIR --model {base,large,mmbert} [--slug {main,exp}] [--smoke] [--epochs N] [--lr X] [--head N] [--tail N] [--budget MIN]` (dùng `argparse`). Ghi vào `OUT_DIR`: notebook + `kernel-metadata.json`.
- Bảng model (giữ trong 1 dict):
  - `base`: HF `FacebookAI/xlm-roberta-base`, source `aisuko/xlm-roberta-base/transformers/default/1`, LR 2e-5
  - `large`: HF `FacebookAI/xlm-roberta-large`, source `aisuko/xlm-roberta-large/transformers/default/1`, LR 1e-5
  - `mmbert`: HF `jhu-clsp/mmBERT-base`, source `aisuko/mmbert-base/transformers/default/1`, LR 3e-5
- Mặc định: `--slug main`, epochs 3, head 384, tail 126, budget 140 (large) / 55 (base, mmbert).
- `slug main` → id `qucanhtrnnguyn/rmit-sgs-lock-in-c1`, title `RMIT_SGS_Lock-In_C1`, code_file `RMIT_SGS_Lock-In_C1.ipynb`. `slug exp` → id `qucanhtrnnguyn/rmit-sgs-lockinexp-c1`, title `RMIT_SGS_LockInExp_C1`, code_file `RMIT_SGS_LockInExp_C1.ipynb` (đổi theo decisions/02).
- Metadata luôn: `is_private: true`, `enable_gpu: true`, `enable_internet: false`, `competition_sources: ["rmit-hackathon-2026"]`, `model_sources: [<source của model>]` (chỉ 1 model), `kernel_type: notebook`, `language: python`.
- Nếu không truyền OUT_DIR dạng thư mục mà là đường dẫn `.ipynb` thì chỉ ghi notebook (để sinh lại bản trong repo).

**S2 – Tìm thư mục model chịu được mọi mirror.** Mirror mmBERT chỉ có `pytorch_model.bin` (không có `model.safetensors`); mirror large có thư mục con `onnx/` cũng chứa `config.json`. Lấy thư mục chứa `config.json` dưới `/kaggle/input`, bỏ đường dẫn chứa `/onnx` hoặc `/.cache`, và thư mục đó phải có `model.safetensors` hoặc `pytorch_model.bin`. Assert tìm được đúng 1 thư mục.

**S3 – Ghi model ở đầu notebook.** Dòng đầu markdown: `# RMIT_SGS_Lock-In_C1` rồi dòng `**Model: <HF name> (mirror <aisuko source>)**`, kèm K, epochs, LR. (Luật yêu cầu ghi rõ model đã chọn ở đầu notebook.)

**S4 – `encode` tổng quát cho mọi tokenizer.** Gọi `tok(list(texts))['input_ids']` (có token đặc biệt, không truncation); nếu `len(x) > HEAD + TAIL + 2` thì giữ `x[:HEAD + 1] + x[-(TAIL + 1):]`. Bỏ việc tự ghép `cls_token_id`/`sep_token_id` (mmBERT dùng token đặc biệt khác).

**S5 – Ngân sách thời gian và thông tin run.**
- `TIME_BUDGET_MIN` lấy từ `--budget`. Đo thời gian từng fold; trước khi bắt đầu fold kế tiếp, nếu `elapsed + thời_gian_fold_trung_bình > TIME_BUDGET_MIN` thì in `STOP EARLY after fold f` và dừng vòng lặp.
- OOF F1, tinh chỉnh trọng số, confusion matrix, F1 theo độ dài/ngôn ngữ chỉ tính trên các dòng đã có OOF. Test = trung bình các fold đã xong (chia cho số fold đã xong, không chia K).
- In ở đầu: `torch.cuda.device_count()`, tên GPU, phiên bản torch/transformers. In sau mỗi fold: `fold f macro F1 ... | train_min ... | fold_min ...` (flush=True).
- Ghi `run_info.json`: model, K, số fold đã xong, stopped_early, epochs, lr, head, tail, oof_raw, oof_tuned, weights, f1 từng fold, phút từng fold, tổng phút, phân bố nhãn dự đoán trên test.
- Assert định dạng submission chỉ chạy khi `not SMOKE` (giữ như cũ).

**S6 – Warmup theo số bước thực.** `steps = ceil(len(ti) / (8 * 2 * max(1, torch.cuda.device_count()))) * EPOCHS`, `warmup_steps = int(0.1 * steps)` (T4 x2 làm batch hiệu dụng gấp đôi).

**S7 – Chịu được transformers 5.x.** Nếu `TrainingArguments` không nhận `group_by_length` (TypeError) thì bỏ tham số đó. Cách gọn: tạo dict tham số, kiểm tra bằng `inspect.signature(TrainingArguments).parameters` rồi lọc. Không dùng API đã bỏ ở v5 (`evaluation_strategy`, `tokenizer=` trong Trainer, `build_inputs_with_special_tokens`).

## Tiêu chí hoàn thành (Sonnet tự kiểm tra rồi báo cáo)

1. Sinh được 6 cấu hình vào thư mục tạm trong scratchpad: `{base,large,mmbert} × {main smoke, exp smoke}` và `large main full`; mỗi `kernel-metadata.json` đúng id/title/code_file/model_sources.
2. Mọi code cell `ast.parse` được (Python 3.13 cú pháp chuẩn).
3. Chạy thử cục bộ phần `encode` + tìm thư mục model với tokenizer giả/thư mục giả là không bắt buộc; nhưng phải chạy được bằng `python -I` một kiểm tra nhỏ cho logic dừng sớm và chuẩn hóa test (ví dụ hàm thuần nhận danh sách thời gian fold) nếu tách được thành hàm.
4. Sinh lại `/home/quocanh/quoc_anh/rmit-hackathon-2026/RMIT_SGS_Lock-In_C1.ipynb` bằng cấu hình `--model large --slug main` (full).
5. Báo cáo: diff tóm tắt, lệnh sinh từng cấu hình, đường dẫn tuyệt đối các thư mục đã sinh. Ghi báo cáo vào `/home/quocanh/quoc_anh/docs/rmit-hackathon-2026/reports/01-sonnet-generator.md`.

## Đợt 2 (theo decisions/02)

**S8 – Đổi tên kernel E.** Sửa `--slug exp` trong `mk_nb.py` thành id `qucanhtrnnguyn/rmit-sgs-lockinexp-c1`, title `RMIT_SGS_LockInExp_C1`, code_file `RMIT_SGS_LockInExp_C1.ipynb`. Sinh lại `scratchpad/gen/*-exp-smoke` và kiểm `kernel-metadata.json`. **Không** đụng thư mục run Haiku đang dùng.
**S9 – Gradient checkpointing chỉ cho large.** Mặc định GC = bật khi `--model large`, tắt với base/mmbert. Thêm cờ `--gc/--no-gc` để ghi đè.
**S10 – Cờ `--no-fp16`.** Mặc định fp16 bật. Cờ này dùng để chạy lại mmBERT nếu fp16 lỗi.
**S11 (chỉ khi Opus gọi) – Notebook probe** `RMIT_SGS_LockInProbe_C1` (`qucanhtrnnguyn/rmit-sgs-lockinprobe-c1`), gồm 1 cell: liệt kê `/kaggle/input` 3 cấp, chạy `nvidia-smi`, in phiên bản python/torch/transformers, nạp tokenizer + config của model đã gắn. Sinh qua `--slug probe`, không cần notebook riêng.
Tiêu chí: các cell parse được, metadata đúng tên. Báo cáo ở `reports/NN-sonnet-s8-s10.md`.

## Đợt 3 (theo plans/01b)

**S12 – Lưu kết quả sau mỗi fold.** Sau mỗi fold, ghi đè `oof.npy`, `test_probs.npy`, `run_info.json` và `submission.csv`. Submission ghi từ các fold đã xong, dùng trọng số đã tinh chỉnh trên OOF đã phủ. Mục đích: nếu `-t` kill kernel giữa chừng thì vẫn còn output. Gom phần tính OOF/trọng số/submission vào 1 hàm, gọi sau mỗi fold và ở cuối; không lặp code. Assert 6000 dòng giữ ở cuối. Ưu tiên: S8 rồi S12, sau đó mới S9/S10. Không sửa thư mục `gen/` mà M1 đang chạy.
