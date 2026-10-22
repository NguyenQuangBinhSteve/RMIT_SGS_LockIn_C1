# Báo cáo Sonnet – S12 (lưu kết quả sau mỗi fold)

Không push, không submit, không git. Không đụng `scratchpad/runs/*` và các thư mục `gen/` cũ.

## Đã làm
- Sửa trên bản sao, kiểm tra xong, chỉ ghi đè `mk_nb.py` sau khi thấy `runs/m1-base` đã tồn tại (10:59). Bản sao đã xóa.
- Gom phần tinh chỉnh trọng số lớp, ghi `oof.npy`, `test_probs.npy` (đã chia số fold xong), `submission.csv`, `run_info.json` vào hàm `save_run(final=False)`. Gọi sau mỗi fold (trước khi xét dừng sớm) và `save_run(True)` sau vòng lặp. `final=True` in thêm confusion matrix, F1 theo độ dài/ngôn ngữ, phân bố nhãn và assert 6000 dòng (khi không SMOKE). Hai cell phân tích/submission cũ đã bỏ, notebook còn 4 cell.
- Mỗi lần gọi tốn thêm 1 lần tìm lưới 36 tổ hợp trên OOF đã phủ (vài giây).

## Kiểm tra
- `ast.parse` mọi code cell; kiểm tra tên chưa định nghĩa: không có.
- `t_logic.py` (nhận đường dẫn notebook làm tham số, mặc định notebook trong repo) có thêm test chạy `save_run` thật trên dữ liệu giả: sau fold 1 và 2 đọc lại `run_info.json` (folds_done, stopped_early=True), `oof.npy`, `test_probs.npy` (tổng xác suất mỗi dòng = 1), `submission.csv`; rồi gọi `final=True`. Cần numpy/pandas/sklearn nên chạy bằng `scratchpad/venv/bin/python -I t_logic.py [notebook]`; bằng `python3 -I` thì phần này tự bỏ qua. Cả hai chạy `ok`.
- Điểm chưa thử: chạy thật trên Kaggle. Nếu `-t` kill đúng lúc đang ghi file thì file cuối có thể dở dang (hiếm; mỗi fold ghi lại nên chỉ mất tối đa phần fold đang ghi).

## File
- Generator: /tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad/mk_nb.py
- Test: /tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad/t_logic.py
- Notebook chính sinh lại (large, main, full): /home/quocanh/quoc_anh/rmit-hackathon-2026/RMIT_SGS_Lock-In_C1.ipynb
- Thư mục mẫu E1 đã dùng S12 (mmbert, exp, budget 100, không smoke): /tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad/gen/mmbert-exp-s12
- M1 base (`runs/m1-base`) do Haiku sinh bằng generator TRƯỚC S12, nên không có lưu theo fold. Nếu muốn M1 có S12 thì sinh lại, nhưng chỉ khi Haiku chưa push.
