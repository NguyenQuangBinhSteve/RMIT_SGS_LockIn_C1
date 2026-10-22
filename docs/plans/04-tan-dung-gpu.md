# Kế hoạch 04 – Tận dụng GPU (11:10 VN)

Nguyên tắc: GPU-giờ còn dư (29.65h) **không phải là giá trị tự thân**. Thứ khan hiếm là thời gian thực đến 17:00 và 6 lượt nộp. Chỉ làm những gì rút ngắn M2 (large, ứng viên nộp chính) hoặc làm nó mạnh hơn, với rủi ro thấp.

## Đánh giá

| # | Ý tưởng | Kết luận | Lý do |
|---|---|---|---|
| 1 | **Mỗi fold một GPU** (2 tiến trình con, `CUDA_VISIBLE_DEVICES=0/1`) thay cho DataParallel | **LÀM**, nhưng có cờ bật/tắt và phải qua smoke trước | Với large (560M tham số), DataParallel phải chép tham số sang GPU thứ 2 ở mọi bước, thường chỉ được ~1.2–1.4× so với 1 GPU. Hai tiến trình độc lập thì gần 2×. Nhưng nếu giữ **K=3** thì vẫn cần 2 vòng (fold 0+1, rồi fold 2 chạy một mình), nên lợi ích nhỏ. Chuyển sang **K=4** (2 vòng × 2 GPU): mỗi fold train trên 75% dữ liệu thay vì 67%, test là trung bình 4 model, thời gian xấp xỉ bằng K=3 với DataParallel. Lợi ích thật là model tốt hơn trong cùng thời gian. |
| 2 | Chia large thành 3 kernel, mỗi kernel 1 fold, rồi kernel M tổng hợp | **BỎ** | (a) Kaggle thường chỉ cho 2 phiên GPU song song, nên kernel thứ 3 sẽ xếp hàng. (b) Notebook M nộp sẽ không chứa phần train, mà dựa vào output của kernel khác, trái tinh thần "Submit notebooks" và khó giải trình. (c) Nhiều điểm hỏng. |
| 3 | Slot E rảnh sau E1 | Dùng **chỉ** để smoke chế độ #1 (nếu chưa smoke xong), sau đó **để trống** | Kết quả ở E không nộp được, và không còn thời gian chạy lại trong M. Thử LR hay độ dài khác cho large ở E không đổi được quyết định nào trước 16:05. E2 (mmBERT 1024) chỉ chạy khi E1 thắng M1 ≥ 0.01, như plans/01b. |
| 4a | Tắt GC + batch 16 không accum cho base/mmBERT | Đã có (S9) | – |
| 4b | Giảm max_len < 512 | **BỎ** | Văn bản dài là vùng khó tách BENIGN/JAILBREAK. group_by_length + dynamic padding đã làm phần lớn batch ngắn. |
| 4c | Predict sắp theo độ dài, eval batch 64 | **LÀM** (rẻ, ít rủi ro) | Predict 6000 test + ~3500 val mỗi fold. Sắp theo độ dài rồi trả về đúng thứ tự giúp tiết kiệm vài phút mỗi fold. |
| 4d | torch.compile, 8-bit optimizer, bf16 | **BỎ** | T4 không có bf16; compile và thư viện mới có rủi ro debug cao, lợi ích không chắc. |

## Thiết kế #1 (giao Sonnet, S13)

- Cờ mới `--parallel-folds` (mặc định tắt). Khi bật: notebook ghi 1 script `fold_worker.py` vào `/kaggle/working`.
- Mã hóa dữ liệu **một lần** ở tiến trình chính, lưu ra đĩa (ids, nhãn, chỉ số fold). Không tokenize lại trong worker.
- Tiến trình chính có hàng đợi fold. Mỗi GPU luôn có 1 worker: `subprocess.Popen` kèm `CUDA_VISIBLE_DEVICES=g`, stdout ghi ra `fold_f.log`. Worker xong thì ghi `oof_f.npy` (chỉ phần val) và `test_f.npy`, rồi chính nó in 1 dòng `fold f macro F1 ... | fold_min ...`. Tiến trình chính in lại dòng đó.
- Ngân sách: trước khi phát fold mới, kiểm `can_continue`. Sau mỗi fold xong, gọi `save_run()` (S12) gộp các file đã có. Mỗi file chỉ do 1 worker ghi, nên không có xung đột.
- Bộ nhớ host: 2 tiến trình large, mỗi cái ~6–8 GB RAM, máy Kaggle ~29 GB, chấp nhận được. Ghi `free -g` vào log đầu run.
- Worker lỗi (returncode ≠ 0): in 40 dòng cuối `fold_f.log` rồi `raise`, để không âm thầm nộp thiếu fold.
- Không cờ thì giữ nguyên đường DataParallel hiện tại. Mọi quy tắc comment như trước.

## Thử nghiệm (giao Haiku)

1. **Ngay khi Sonnet xong S13:** smoke chế độ song song bằng **kernel probe** `RMIT_SGS_LockInProbe_C1` (`--slug probe --model large --smoke --parallel-folds`, K=2, `-t 1500`). Cách này đồng thời kiểm luôn **giới hạn số phiên GPU**: nếu kernel bị QUEUED trong lúc M1 và E1 đang chạy thì giới hạn là 2. Ghi lại và chờ, không hủy gì. Kernel probe đúng quy ước tên và không nộp.
2. Đạt khi: COMPLETE, cả 2 fold có log, `oof.npy` phủ toàn bộ, mỗi worker in đúng GPU riêng (`torch.cuda.current_device` + tên). So `fold_min` với smoke large cũ (DataParallel) và báo cáo.
3. Báo cáo: `docs/reports/NN-probe-parallel.md`.

## Thay đổi M2 / E2

- **M2 (large, M), push ngay khi M1 xong. Không chờ chế độ song song quá M1 xong + 10 phút.**
  - Nếu smoke song song đạt: `--model large --slug main --parallel-folds` với **K=4**, 3 epoch, budget B theo plans/01b. Quy tắc epoch: 1 vòng ≈ 3.5 × t_base × 1.1 (vì 75% dữ liệu và 1 GPU/fold, so với t_base đo bằng DataParallel; ước lượng thận trọng). Nếu 2 vòng × giá trị đó ≤ B thì 3 epoch, không thì 2 epoch.
  - Nếu chưa đạt: giữ đúng plans/01b (DataParallel, K=3).
- Cổng: OOF K=4 được so trực tiếp với M1 (K=3). Chênh lệch do K là có lợi thật cho model nộp, nên vẫn dùng ngưỡng +0.003.
- **E2:** không đổi, theo plans/01b. E1 xong mà không thắng thì để slot E trống.
- M1/E1 đang chạy: không đụng.

## Việc giao Sonnet

- **S13:** cờ `--parallel-folds` và `--k` (mặc định 3), theo thiết kế trên. Thêm vào `tools/t_logic.py` 1 kiểm tra cho hàng đợi fold/ngân sách nếu tách được thành hàm thuần.
- **S14:** predict sắp theo độ dài (4c), eval batch 64.
- Thêm `--slug probe` nếu chưa có (S11).
- Báo cáo: `docs/reports/NN-sonnet-s13-s14.md`. Hạn: 11:45 VN.
