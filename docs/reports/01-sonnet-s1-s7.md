# Báo cáo Sonnet – S1 đến S7 (sửa generator notebook)

File đã sửa: `/tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad/mk_nb.py` (viết lại, vẫn một file).
Notebook chính đã sinh lại: `/home/quocanh/quoc_anh/rmit-hackathon-2026/RMIT_SGS_Lock-In_C1.ipynb` (large, main, SMOKE=False).
Test logic: `/tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad/t_logic.py`.

## Lệnh sinh
```
S=/tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad
python3 $S/mk_nb.py OUT_DIR --model {base,large,mmbert} [--slug main|exp] [--smoke] [--epochs N] [--lr X] [--head N] [--tail N] [--budget MIN]
python3 $S/mk_nb.py /home/quocanh/quoc_anh/rmit-hackathon-2026/RMIT_SGS_Lock-In_C1.ipynb --model large --slug main   # chỉ ghi notebook
```
Đã sinh sẵn 6 cấu hình trong `$S/gen/`: `{base,large,mmbert}-{main,exp}-smoke` và `large-main-full` (mỗi thư mục có notebook + `kernel-metadata.json`). Đường dẫn đầy đủ: `/tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad/gen/<tên>/`.

## S1 Tham số dòng lệnh
argparse theo đúng đặc tả; bảng `MODELS` (HF, source, LR, budget mặc định 140 large / 55 còn lại) và `SLUGS` là hai dict. Metadata đủ các trường yêu cầu, `model_sources` chỉ 1 model. OUT kết thúc bằng `.ipynb` thì chỉ ghi notebook.
Kiểm tra: đọc lại 7 `kernel-metadata.json` (6 cấu hình + bản trong gen), id/title/code_file/model_sources đúng. Không chắc: không có.

## S2 Tìm thư mục model
`find_model(root)` lấy các thư mục chứa `config.json`, bỏ đường dẫn có `/onnx` hoặc `/.cache`, giữ thư mục có `model.safetensors` hoặc `pytorch_model.bin`, assert đúng 1.
Kiểm tra: `python -I t_logic.py` trên cây thư mục giả (chỉ .bin; có onnx và .cache cùng config; 0 kết quả thì assert). Đã xem danh sách file thật của 3 mirror bằng kaggle CLI (mmBERT chỉ có `pytorch_model.bin`, large có `onnx/config.json`): khớp. Không chắc: đường dẫn thực dưới `/kaggle/input` chưa thử trên Kaggle (glob `**` mặc định bỏ qua thư mục ẩn, thêm lớp lọc `.cache` cho chắc).

## S3 Ghi model ở đầu notebook
Markdown: `# RMIT_SGS_Lock-In_C1`, dòng `**Model: <HF> (mirror <aisuko/...>)**`, dòng `K=3, epochs=.., LR=..`. Mirror ghi dạng ngắn `aisuko/xlm-roberta-large` (bỏ hậu tố `/transformers/default/1`) theo ví dụ ở plan 01; nếu muốn đủ hậu tố thì sửa biến `mirror` trong mk_nb.py. Bản smoke ghi thêm "(smoke run)"; header ghi K=3/epochs gốc dù smoke dùng K=2, epochs=1.

## S4 encode
`tok(list(texts))['input_ids']`, nếu `len(x) > HEAD+TAIL+2` thì `x[:HEAD+1] + x[-(TAIL+1):]`. Đã bỏ cls/sep thủ công. Kiểm tra bằng tokenizer giả trong t_logic.py (độ dài, giữ đầu/cuối). Không chắc: tokenizer thật phát cảnh báo "sequence longer than max length" (vô hại).

## S5 Ngân sách thời gian và run_info
- `can_continue(elapsed, fold_mins, budget)` là hàm thuần; sau mỗi fold (nếu còn fold) kiểm tra, in `STOP EARLY after fold f` rồi break. `elapsed` tính từ đầu notebook (T0, gồm cả thời gian nạp). Test: 3 trường hợp trong t_logic.py.
- OOF, tinh chỉnh, confusion matrix, F1 theo độ dài/ngôn ngữ chỉ dùng dòng có `mask`; test `pte /= số fold đã xong`.
- In số GPU, tên GPU, torch/transformers ở đầu; in `fold f macro F1 .. | train_min .. | fold_min ..` flush.
- `run_info.json` có đủ các trường yêu cầu (thêm mirror, folds_done, test_pred_dist).
- Assert submission chỉ khi `not SMOKE`.
Không chắc: chưa chạy end to end (không có torch cục bộ); mới kiểm tra cú pháp và các hàm thuần. `fold_f1`/`oof_*` là float numpy, `json.dump` chấp nhận float64 nhưng nếu lỗi kiểu thì cast `float()`; (np.float64 là subclass của float nên ổn).

## S6 Warmup
`steps = ceil(len(ti) / (8*2*max(1, device_count))) * EPOCHS`, `warmup_steps = int(0.1*steps)`.

## S7 transformers 5.x
Dict `kw` rồi lọc theo `inspect.signature(TrainingArguments).parameters` (lọc mọi khóa, không chỉ `group_by_length`; ngắn hơn và tránh lỗi tương tự). Không dùng `evaluation_strategy`, `tokenizer=`, `build_inputs_with_special_tokens`. Giữ `processing_class`, `warmup_steps`.

## Điểm rủi ro cần Haiku để ý khi smoke
1. mmBERT (ModernBERT) với `fp16=True` và `gradient_checkpointing` chưa thử; nếu NaN loss hoặc lỗi thì báo Opus (có thể cần bf16 không có trên T4, hoặc tắt checkpointing).
2. `pytorch_model.bin` của mmBERT cần torch đủ mới để `torch.load` (Kaggle thường đủ).
3. `compute_loss` giữ `**kw` nên chịu `num_items_in_batch` của v5.
4. Đã sinh lại notebook chính từ cấu hình large/main/full; cấu hình cũ (hardcode xlm-roberta-large) tương đương về thuật toán.
