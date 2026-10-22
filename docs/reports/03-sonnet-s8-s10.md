# Báo cáo Sonnet – S8 đến S10 (đợt 2)

Generator: `/tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad/mk_nb.py`. Không push, không submit, không commit. S11 chưa làm (đúng chỉ thị).

## Thay đổi
- S8: `--slug exp` thành id `qucanhtrnnguyn/rmit-sgs-lockinexp-c1`, title `RMIT_SGS_LockInExp_C1`, code_file `RMIT_SGS_LockInExp_C1.ipynb`.
- S9: `--gc/--no-gc` (argparse.BooleanOptionalAction). Không truyền thì GC chỉ bật khi `--model large`. Nạp vào hằng `GC` ở cell 1, dùng ở `gradient_checkpointing=GC`.
- S10: `--no-fp16` đặt hằng `FP16=False`, dùng ở `fp16=FP16`. `run_info.json` ghi thêm `gc`, `fp16`.

## Kiểm tra
- Mọi code cell của các thư mục mới `ast.parse` được; metadata đúng tên/id/code_file/model_sources.
- GC mặc định: base False, mmbert False, large True (đọc từ hằng trong notebook sinh ra); `--no-gc` cho large cho False; `--no-fp16` cho FP16=False.
- `python3 -I t_logic.py` vẫn in `ok`.
- Notebook chính large/main/full sinh lại: diff so bản cũ chỉ gồm 3 chỗ (thêm GC/FP16 vào dòng cấu hình, `gradient_checkpointing=GC`, `fp16=FP16`, thêm gc/fp16 vào run_info). Với large GC=True, FP16=True nên hành vi không đổi.

## Thư mục (không đụng thư mục cũ)
Chưa có `reports/02-haiku-smoke.md`, không chắc Haiku đã push xong, nên KHÔNG ghi đè `gen/*-exp-smoke`, `large-main-smoke`, `mmbert-exp-smoke` hay các thư mục main-smoke khác (các thư mục này vẫn là bản cũ: GC bật cho mọi model, kernel exp tên cũ `lockin-c1-exp`). Thư mục mới, tất cả trong `/tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad/gen/`:
- `base-exp-smoke-ren`, `large-exp-smoke-ren`, `mmbert-exp-smoke-ren` (smoke, tên kernel mới, GC mặc định mới)
- `mmbert-exp-nofp16` (full, `--no-fp16 --budget 75`, dùng khi mmBERT lỗi fp16)
- `large-exp-nogc` (full, `--no-gc`, tham khảo)
Lưu ý: `*-main-smoke` cũ vẫn bật GC cho base/mmbert; nếu cần smoke main theo mặc định mới thì sinh thư mục mới (ví dụ `python3 mk_nb.py gen/base-main-smoke-v2 --model base --slug main --smoke`).
