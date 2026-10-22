# Kế hoạch 01b – Lịch sửa sau smoke (11:05 VN). Thay phần "Lịch" và "Điều kiện dừng" của plans/01

## Đường dẫn mới (áp dụng từ 11:10 VN, thay mọi đường dẫn scratchpad hoặc docs/rmit-hackathon-2026 trong các plan cũ)

```bash
R=/home/quocanh/quoc_anh/LockIn_RMIT_HACKATHON_2026
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token); K=$R/.venv/bin/kaggle; PY=$R/.venv/bin/python
M=qucanhtrnnguyn/rmit-sgs-lock-in-c1; E=qucanhtrnnguyn/rmit-sgs-lockinexp-c1; P=qucanhtrnnguyn/rmit-sgs-lockinprobe-c1
```
- Generator: `$R/tools/mk_nb.py` (test: `$R/tools/t_logic.py`). Notebook nộp: `$R/notebooks/`.
- Thư mục run: `$R/runs/<RUN>/` (sinh bằng `$PY $R/tools/mk_nb.py $R/runs/<RUN> ...`), output tải về `$R/runs/<RUN>/out`.
- Tài liệu: `$R/docs/{plans,decisions,reports,reference}`. Không ghi ra ngoài `$R`. Không git commit/push.

## Nhận định về số đo smoke

- Ngoại suy 70× **phóng đại quá nhiều**. Mỗi fold smoke chỉ có khoảng 13 bước (400 dòng / batch hiệu dụng 32), nên 1.25 phút chủ yếu là chi phí cố định (khởi tạo Trainer, warmup CUDA, đưa model lên GPU). Ước lượng theo thông lượng (1.3M token/epoch/fold, T4 x2 fp16) ra: base khoảng 3–6 phút/epoch, large khoảng 10–20 phút/epoch. Chỉ tin **số đo fold 0 của bản full**.
- Smoke large không có dấu hiệu sụp: dự đoán ra đủ 3 lớp, 10 PROMPT_INJECTION đúng. OOF thấp là do 400 dòng × 1 epoch với LR 1e-5.
- ERROR với log trống là lỗi tạm thời phía Kaggle (v4 và v5 cùng nội dung). Gặp lỗi này thì push lại 1 lần.

## Thông số

| Run | Kernel | Lệnh sinh | Budget | `-t` | Push |
|---|---|---|---|---|---|
| M1 | M | `--model base --slug main --budget 100` (K=3, 3 epoch, LR 2e-5) | 100 | 7200 | **Ngay** |
| E1 | E (tên mới) | `--model mmbert --slug exp --budget 100` (LR 3e-5) | 100 | 7200 | Ngay sau khi Sonnet xong S8 (đổi tên E). Không push E dưới tên cũ nữa. |
| M2 | M | `--model large --slug main --epochs E2 --budget B` | B = phút còn lại đến 16:05 − 10 | B×60 + 900 | Ngay khi M1 xong |
| E2 | E | Chỉ khi E1 OOF ≥ M1 OOF + 0.01: `--model mmbert --slug exp --head 768 --tail 254 --budget 90` | 90 | 6300 | Sau E1 |
| M3 | M | Bỏ, trừ khi M2 xong trước 14:30 | – | – | – |

**Cách chọn epoch cho M2.** Đọc `fold_min` của fold 0 trong M1 (`kernels logs`), gọi là t_base. Ước lượng 1 fold large 3 epoch ≈ 3.5 × t_base, vì large nhiều tham số gấp khoảng 3 lần và có thêm gradient checkpointing.
- Nếu 3 × 3.5 × t_base ≤ B thì dùng `--epochs 3`.
- Nếu 3 × 2.4 × t_base ≤ B thì dùng `--epochs 2`.
- Nếu không thì dùng `--epochs 2` và chấp nhận dừng sớm sau 2 fold.
- Haiku tính số này rồi báo Opus duyệt trước khi push M2.

## Lịch (giờ VN)

| Giờ | M | E |
|---|---|---|
| 11:10 | Push M1 (base) | Sonnet làm S8–S10 + S12 |
| ~11:25 | Đọc fold 0 của M1 → báo t_base | Push E1 (mmBERT) khi S8 xong |
| ~11:55–12:20 | M1 xong → Gate A → nộp #1 | E1 chạy |
| 12:20–16:05 | M2 large (budget theo công thức) | E1 xong ~12:40 → quyết E2 |
| 16:05–16:30 | Gate B → nộp #2 | – |
| 16:30–17:00 | Chọn bài cuối theo decisions/02 Q3. Không push run mới sau 16:25. | – |

## Sửa luật cổng cho run dừng sớm

Run M2 dừng sớm nhưng đã xong ≥ 2 fold vẫn được xét nộp. Điều kiện: OOF trên các dòng đã phủ ≥ OOF tốt nhất đã nộp + 0.005, và assert định dạng 6000 dòng đạt. Run chỉ xong 1 fold thì không nộp.

## Điều kiện quay lại Opus

- Fold 0 của M1 có F1 < 0.8, hoặc fold 0 của M2 có F1 < 0.6.
- M1 chưa xong lúc 12:30.
- ERROR với log trống 2 lần liên tiếp trên cùng run.
- E1 OOF > M1 OOF + 0.02. Khi đó cân nhắc thay large bằng mmBERT trong M.
