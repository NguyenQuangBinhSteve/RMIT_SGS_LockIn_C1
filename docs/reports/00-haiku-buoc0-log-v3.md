# Báo cáo 00 – Bước 0: chẩn đoán smoke v3 (không GPU)
- Giờ kiểm tra (VN): 10:42–10:45, 22/10/2026 | Kernel: qucanhtrnnguyn/rmit-sgs-lock-in-c1 version 3 (lastRunTime 03:32:26 UTC = 10:32 VN)
- Lệnh đã chạy (không push, không submit, không git, không sửa code):
  - `kernels status` (kernel gốc và `.../3`): `KernelWorkerStatus.ERROR`
  - `kernels output -p .../out3b`: tải về `rmit-sgs-lock-in-c1.log`, nội dung đúng 2 byte `[]`, không có file output nào khác
  - `kernels logs` (lặp lại 2 lần, cách nhau vài phút): `[]`
  - `kernels list --mine -v`: chỉ có 1 kernel, 0 vote
  - `kernels pull -m -p .../pulled3`: lấy được `kernel-metadata.json` và `.ipynb` của bản mới nhất
  - `kernels output/logs/status` cho version 1 và 2: đều ERROR, log cũng `[]`
  - `kernels pull` cho version 1 và 2: 403 Forbidden (chỉ pull được bản mới nhất)
  - `models instances versions files aisuko/xlm-roberta-large/Transformers/default/1`: liệt kê file mô hình
  - `competitions download rmit-hackathon-2026 -f train.csv / test.csv` (chỉ đọc dữ liệu): kiểm schema
- Trạng thái cuối: **ERROR** (v1, v2, v3 đều ERROR). stopped_early: không xác định (không có log). Folds xong: 0/? (không có log, không có output).
- GPU: không có thông tin (không có log). Metadata yêu cầu `NvidiaTeslaT4`, `enable_gpu: true`, `enable_internet: false`.
- torch/transformers: không có thông tin.
- OOF macro F1, F1 theo fold, phút theo fold, phân bố nhãn test, submission.csv: **không có** (không có output).
- GPU-giờ: `kaggle quota` lúc kiểm tra: GPU used 0.01h / remaining 29.99h / total 30.00h (refresh 2026-10-24 00:00). Bước 0 không chạy GPU nên số này là trước và sau như nhau. TPU 0.00h / 20.00h.
- (smoke) Ngoại suy phút/fold full: không tính được (không có số phút).

## Lỗi thấy được
- Không có traceback nào. Log trả về `[]` cho cả 3 version, nên không có nguyên nhân từ log.
- Metadata v3 (`pulled3/kernel-metadata.json`): `competition_sources: rmit-hackathon-2026`, `model_sources: aisuko/xlm-roberta-large/Transformers/default/1`, `machine_shape: NvidiaTeslaT4`, `enable_internet: false`, `enable_gpu: true`. Không thấy trường nào bất thường.
- Notebook `.ipynb` mở được bằng json (nbformat 4, 5 cell code/markdown, không có output lưu sẵn).

## Đã kiểm chứng tĩnh (không chạy notebook)
- Mô hình: `aisuko/xlm-roberta-large/Transformers/default/1` có `model.safetensors` ở thư mục gốc (2.24 GB) và `config.json`, `tokenizer.json`, `sentencepiece.bpe.model`. Glob trong cell 1 `/kaggle/input/**/xlm-roberta-large/**/model.safetensors` khớp với đường dẫn này. Có `pytorch_model.bin` nên cũng không lo thiếu weight.
- Dữ liệu: competition có `train.csv`, `test.csv`, `sample_submission.csv`. Schema khớp notebook:
  - train: 14000 dòng, cột `id,text,language,label`. Nhãn: BENIGN 8680, JAILBREAK 4340, PROMPT_INJECTION 980 (7.0%). Không có nhãn lạ, không có text rỗng, không trùng id.
  - test: 6000 dòng, cột `id,text,language`. Không có text rỗng.
- Kết luận: đường dẫn dữ liệu và mô hình, cột và nhãn đều khớp với code. Không tìm thấy lỗi code rõ ràng trong 5 cell. Chưa loại trừ được lỗi xảy ra trước khi cell 1 chạy (gắn nguồn, mount, môi trường Docker, tài nguyên), vì không có log để xác minh.

## Kết luận nguyên nhân
- **Chưa xác minh.** Không có log hay traceback từ Kaggle nên không biết lỗi xảy ra ở cell nào.
- Cả v1 và v2 cũng ERROR với log trống. Có thể là lỗi hệ thống (môi trường hoặc cấu hình kernel) chứ không riêng một bản code. Không xác nhận được vì không pull được v1, v2.

## Đề xuất (chưa làm, cần Opus quyết)
1. Không push lại v3 để thử: Bước 0 không cho phép và lần thử lại không đổi gì sẽ chỉ lặp lỗi. Log `[]` đã được thử lại nhiều lần.
2. Nếu cần chẩn đoán tiếp, đề xuất một kernel probe rất nhỏ: chỉ `print(os.listdir('/kaggle/input'))`, `nvidia-smi`, `import torch, transformers`, `-t` ngắn (vài phút), cùng cấu hình gắn nguồn. Mục đích là kiểm xem Kaggle có ghi log không và cell đầu có chạy không. Chạy probe tốn GPU ít, nhưng cần Opus duyệt.
3. Venv CLI (`/tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad/venv/bin/python`) chưa có pandas, torch, transformers. Hệ thống `/usr/bin/python3` cũng chưa có pandas. Bước 1 (`mk_nb.py`) và Bước 3 (đọc `submission.csv`, `oof`) sẽ lỗi nếu dùng `$PY` như runbook. Cần cài pandas vào một Python dùng được trước Bước 3.
4. Không có GPU-giờ nào bị tiêu. Hạn mức còn lại 29.99h.

File tạm trong scratchpad: `.../scratchpad/out3b/rmit-sgs-lock-in-c1.log`, `.../scratchpad/pulled3/` (metadata và notebook v3), `.../scratchpad/data_check/` (train.csv, test.csv).
