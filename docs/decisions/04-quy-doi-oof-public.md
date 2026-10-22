# Quyết định 04 – Quy đổi OOF sang public và cổng mới (11:25 VN)

Căn cứ: `docs/reports/09-tfidf-calib.md`. TF-IDF OOF group-fold 0.9168, StratifiedKFold 0.9165, public của bài cùng loại 0.91819.

## Kết luận
- **Public dự kiến ≈ OOF group-fold**, độ lệch ~0 (+0.001, thô). Giả định "OOF thấp hơn public do prefix trùng" bị bác: chia group hay không chia gần như cùng điểm.
- Giữ StratifiedGroupKFold (không tốn gì, an toàn hơn).

## Cổng (thay decisions/03 mục a)
| Cổng | Điều kiện | Hành động |
|---|---|---|
| Gate A (M1 base) | OOF ≥ **0.93** và định dạng đạt | Nộp #1 transformer |
| Kiểm pipeline | OOF M1 < 0.9168, tức thua TF-IDF | Không nộp. Báo Opus kèm confusion matrix và F1 theo độ dài/ngôn ngữ. Nghi lỗi pipeline (cắt token, trọng số lớp, LR). |
| Vùng xám | 0.9168 ≤ OOF M1 < 0.93 | Không nộp base. Dồn lượt cho M2. |
| Gate B (M2…) | OOF ≥ OOF tốt nhất đã nộp + 0.003 **và** ≥ 0.93 | Nộp |

- Mục tiêu: **OOF ≥ 0.97** (≈ public top 10). Mốc chấp nhận được: ≥ 0.95.
- Nếu M1 base < 0.95 thì M2 large phải dùng 3 epoch nếu thời gian cho phép. Ưu tiên epoch hơn K=4: nếu không đủ thời gian cho cả hai thì chọn K=3 song song với 3 epoch, bỏ K=4 với 2 epoch.
- Còn 5 lượt; giữ ≥ 2 dự phòng.
