# Báo cáo Sonnet – S13, S14 và slug probe

Chỉ ghi trong thư mục dự án. Không đụng `runs/`, không push, không submit, không git. Lệnh kiểm tra: `.venv/bin/python -I tools/t_logic.py` (cần numpy/pandas/sklearn, in `ok`, `save_run ok`, `run_parallel ok`).

## Thay đổi trong `tools/mk_nb.py`
- Cờ mới: `--k` (mặc định 3) và `--parallel-folds` (mặc định tắt). `K` trong header và cấu hình lấy từ `--k`.
- `--slug probe`: id `qucanhtrnnguyn/rmit-sgs-lockinprobe-c1`, title `RMIT_SGS_LockInProbe_C1`, code_file `RMIT_SGS_LockInProbe_C1.ipynb`, private. Là cùng một notebook với exp, chỉ khác tên kernel (đúng cách dùng ở plans/04: `--slug probe --model large --smoke --parallel-folds`).
- Refactor cell huấn luyện: huấn luyện một fold nằm trong hàm `train_fold(ds_tr, ds_va, ds_te)` (cùng `softmax`, `WeightedTrainer`, `predict_probs`), gộp kết quả bằng `add_fold`, `save_run` giữ nguyên (thêm trường `parallel` trong run_info). Đường tuần tự và đường song song dùng chung các hàm này; đoạn mã dùng chung được sinh một lần trong generator và chèn vào cả notebook lẫn `fold_worker.py`.
- Cấu hình không có cờ mới sinh ra notebook tương đương về hành vi (cùng K=3, DataParallel, cùng log `fold f macro F1 | train_min | fold_min`), nhưng mã đã được tổ chức lại nên văn bản notebook khác bản cũ. `notebooks/RMIT_SGS_Lock-In_C1.ipynb` đã sinh lại (large, main, full). `runs/m1-base` và `runs/e1-mmbert` KHÔNG đổi, vẫn là bản cũ.

## S13 – `--parallel-folds`
- Tokenize một lần ở tiến trình chính; `run_parallel` lưu `pf/ds_all`, `pf/ds_te` (save_to_disk) và `pf/job.json` (tham số, trọng số lớp, chỉ số train/val của mọi fold), ghi `fold_worker.py` từ chuỗi `WORKER_SRC`, in `free -g`.
- Mỗi GPU g (`torch.cuda.device_count()`, tối thiểu 1) chạy một worker `subprocess.Popen([python, fold_worker.py, f], env CUDA_VISIBLE_DEVICES=g, stdout -> pf/fold_f.log)`. Hàng đợi fold; trước khi phát fold mới gọi `next_fold` (fold đầu tiên luôn phát, sau đó dùng `can_continue`). Hết ngân sách thì in `STOP EARLY after N folds` và xóa hàng đợi, các fold đang chạy vẫn được chờ.
- Worker chỉ ghi `pf/oof_f.npy` (phần val) và `pf/test_f.npy`, in GPU của mình (CUDA_VISIBLE_DEVICES, `current_device`, tên GPU) và `train_min` vào log. Tiến trình chính nạp file, gọi `add_fold` (in `fold f macro F1 .. | fold_min ..`, không có `train_min` vì main không đo) rồi `save_run()`; cuối cùng `save_run(True)`.
- Worker lỗi: in 40 dòng cuối `pf/fold_f.log`, kill các worker còn chạy, `raise RuntimeError`.
- Trong worker mỗi tiến trình thấy 1 GPU nên `steps` tính theo 1 GPU, batch hiệu dụng 16 (DataParallel là 32).
- Test trong `tools/t_logic.py`: `next_fold` (hàm thuần), và `run_parallel` thật chạy với worker giả (2 GPU giả, K=4): đủ 4 fold, GPU gán 0,0,1,1 theo từng worker, OOF phủ đủ, test = trung bình; fold lỗi gây `RuntimeError`; ngân sách nhỏ dừng sau 2 fold. Source của `fold_worker.py` thật cũng được `ast.parse`.

## S14
`predict_probs` sắp theo độ dài tăng dần rồi trả về đúng thứ tự ban đầu (`np.argsort`, gán lại bằng chỉ số); `per_device_eval_batch_size=64`. Áp dụng cho cả đường tuần tự và song song.

## Chưa kiểm chứng (cần smoke bằng kernel probe)
- Toàn bộ đường torch/transformers chưa chạy thật cục bộ (không có torch). Đã kiểm tra: ast.parse, không có tên chưa định nghĩa, logic hàng đợi/subprocess/gộp bằng test.
- `Dataset.select(list)` trên dataset đã `select`, `save_to_disk`/`load_from_disk`, và worker `import` đầy đủ chỉ chạy được trên Kaggle.
- Tiến trình chính khởi tạo CUDA ở GPU 0 (do `get_device_name`) tốn khoảng 300 MB; không ảnh hưởng đáng kể.
- Thư mục `pf/` (dataset, npy, log) nằm trong `/kaggle/working` nên vào output của kernel (nhỏ).
- Chạy lệnh test bằng `python3 -I` (không phải `.venv`) sẽ lỗi vì thiếu numpy.

## Lệnh mẫu
```
R=/home/quocanh/quoc_anh/LockIn_RMIT_HACKATHON_2026
python3 $R/tools/mk_nb.py OUT --model large --slug probe --smoke --parallel-folds        # probe smoke (K=2)
python3 $R/tools/mk_nb.py OUT --model large --slug main --parallel-folds --k 4           # M2 song song
```
