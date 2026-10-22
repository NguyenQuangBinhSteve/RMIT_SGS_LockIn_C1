# Kế hoạch 03 – Runbook cho Haiku 5.5 (high): chạy thực nghiệm trên Kaggle

Lịch và cổng: `plans/01-ke-hoach-gpu.md`. Quyết định: `decisions/01-quyet-dinh.md`.

## Cấm tuyệt đối

- KHÔNG chạy `kaggle competitions submit`. KHÔNG `git commit`/`git push`.
- KHÔNG push vào kernel đang `RUNNING`/`QUEUED` (kiểm `status` trước mỗi lần push).
- KHÔNG sửa `mk_nb.py` hay notebook; lỗi code thì báo lại để Sonnet sửa.
- Sau 16:25 VN không push run mới.

## Biến môi trường (dán đầu mỗi lệnh Bash)

```bash
S=/tmp/claude-1000/-home-quocanh-quoc-anh/fbb51667-0d23-4bf5-a5e7-8c31a08f1d4e/scratchpad
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token); K=$S/venv/bin/kaggle; PY=$S/venv/bin/python
M=qucanhtrnnguyn/rmit-sgs-lock-in-c1; E=qucanhtrnnguyn/rmit-sgs-lockinexp-c1   # tên mới, xem decisions/02
```

## Bước 0 – Chẩn đoán smoke v3 (không tốn GPU, làm ngay)

```bash
$K kernels status $M
$K kernels output $M -p $S/out3b     # tải log + file output
$K kernels logs $M | head -c 3000
```
Log vẫn `[]` thì ghi nhận và chuyển sang Bước 1 sau khi Sonnet xong (bản mới sẽ thay v3).

## Bước 1 – Sinh và push một run

Tên run (RUN): `m-smoke`, `e-smoke`, `m1-base`, `e1-mmbert`, `m2-large`, `e2-mmbert1024`, `m3-...`.

```bash
$PY $S/mk_nb.py $S/runs/RUN --model large --slug main --smoke          # ví dụ m-smoke
cat $S/runs/RUN/kernel-metadata.json                                     # kiểm id, model_sources, enable_internet=false
$K kernels status $M                                                      # phải KHÔNG phải RUNNING/QUEUED
$K kernels push -p $S/runs/RUN --accelerator NvidiaTeslaT4 -t SECONDS
```
Tham số theo lịch:

| RUN | Lệnh sinh | `-t` |
|---|---|---|
| m-smoke | `--model large --slug main --smoke` | 1500 |
| e-smoke | `--model mmbert --slug exp --smoke` | 1500 |
| m1-base | `--model base --slug main --budget 55` | 4200 |
| e1-mmbert | `--model mmbert --slug exp --budget 55` | 4200 |
| m2-large | `--model large --slug main --budget 140` (thêm `--epochs 2` nếu Opus quyết) | 9300 |
| e2-mmbert1024 | `--model mmbert --slug exp --head 768 --tail 254 --budget 90` (chỉ khi Opus duyệt) | 6300 |

Ghi lại giờ push (giờ VN) và số version Kaggle in ra.

## Bước 2 – Theo dõi

Không dùng `sleep` ở foreground. Chạy vòng poll ở chế độ nền (`run_in_background: true`) và chờ thông báo:
```bash
while true; do s=$($K kernels status SLUG 2>&1); echo "$(date +%H:%M) $s"; case "$s" in *RUNNING*|*QUEUED*) sleep 60;; *) break;; esac; done
```
Xem tiến độ giữa chừng (in các dòng `fold ... macro F1`):
```bash
$K kernels logs SLUG > $S/runs/RUN/live.log; $PY -I -c "import json,sys; [print(x['data'],end='') for x in json.load(open(sys.argv[1]))]" $S/runs/RUN/live.log | grep -E "fold|STOP|Error|cuda|OOF" | tail -20
```

## Bước 3 – Lấy kết quả

```bash
$K kernels output SLUG -p $S/runs/RUN/out
ls -la $S/runs/RUN/out; cat $S/runs/RUN/out/run_info.json
$PY -I -c "import json,sys; [print(x['data'],end='') for x in json.load(open(sys.argv[1]))]" $S/runs/RUN/out/*.log | tail -60
$PY -I -c "import pandas as pd,sys; s=pd.read_csv(sys.argv[1]); print(len(s), list(s.columns)); print(s.label.value_counts())" $S/runs/RUN/out/submission.csv
$K quota
```
Cần đọc: OOF raw/tuned, F1 từng fold, phút từng fold, `stopped_early`, số GPU (`device_count`), phân bố nhãn test (PROMPT_INJECTION nên khoảng 5–9%).
Với smoke: tính **phút train 1 fold full ≈ 70 × phút train 1 fold smoke** và ghi vào báo cáo.

## Khi log trống hoặc ERROR

1. Đợi 3 phút rồi chạy lại `kernels output` và `kernels logs` (log có thể đồng bộ chậm).
2. Vẫn trống: chạy `$K kernels pull SLUG -p $S/runs/RUN/pulled -m` và so `kernel-metadata.json` (model_sources, machine_shape, enable_internet) với bản đã push; kiểm notebook mở được: `$PY -I -c "import json,sys; json.load(open(sys.argv[1]))" <ipynb>`.
3. Push lại **1 lần** không đổi gì. Lỗi log trống lần 2 → dừng, báo Opus.
4. Log có traceback → chép 30 dòng cuối traceback vào báo cáo, báo Sonnet/Opus. Không tự sửa code.

## Báo cáo

Mỗi run 1 file: `/home/quocanh/quoc_anh/docs/rmit-hackathon-2026/reports/NN-<RUN>.md` (NN tăng dần, tiếp số sau file cuối trong thư mục). Mẫu:

```markdown
# Báo cáo NN – <RUN>
- Giờ push / giờ xong (VN): ... / ...  | Kernel: <slug> version <N>
- Lệnh đã chạy: (sinh, push với -t)
- Trạng thái cuối: COMPLETE / ERROR / CANCEL; stopped_early: có/không (folds xong k/K)
- GPU: device_count, tên GPU; torch/transformers version
- OOF macro F1: raw ... | tuned ... | weights ...
- F1 từng fold: ...  | phút từng fold: ... | tổng phút: ...
- F1 theo độ dài / ngôn ngữ: ...
- Phân bố nhãn test: ...
- submission.csv: số dòng, cột (đạt/không đạt định dạng)
- GPU-giờ: `kaggle quota` trước ... / sau ...
- (smoke) Ngoại suy phút/fold full: ...
- Lỗi: (traceback rút gọn nếu có)
- Đề xuất: (vd qua Gate A/B? cần Opus quyết gì?)
```
Sau mỗi báo cáo, chuyển cho Opus quyết định bước tiếp; đặc biệt báo ngay khi gặp điều kiện dừng trong `plans/01-ke-hoach-gpu.md`.
