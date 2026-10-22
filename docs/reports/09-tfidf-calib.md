# Báo cáo 09 – Hiệu chuẩn TF-IDF + LogisticRegression (OOF)

Script: `R/tools/tfidf_calib.py` (chỉ để hiệu chuẩn, không nộp). Dữ liệu: `R/data/train.csv` (14000 dòng), tải bằng `kaggle competitions download rmit-hackathon-2026` và giải nén vào `R/data`.

## Cấu hình chạy
- Đặc trưng: FeatureUnion gồm char TF-IDF (ngram 2–5, min_df=2, sublinear_tf) và word TF-IDF (ngram 1–2, min_df=2, sublinear_tf).
- Mô hình: LogisticRegression(C=10, class_weight='balanced', max_iter=300).
- 3 fold (`n_splits=3`, shuffle, random_state=42).
- Nhóm (chỉ dùng cho StratifiedGroupKFold): văn bản lowercase, thay ký tự không phải chữ/số bằng khoảng trắng, cắt 60 ký tự đầu (prefix-60). Có 13569 nhóm trên 14000 dòng.

## Kết quả

| Splitter | OOF macro F1 | Thời gian |
|---|---|---|
| StratifiedGroupKFold (prefix-60) | **0.9168** | 1.4 phút |
| StratifiedKFold (thường) | **0.9165** | 1.1 phút |

Tổng thời gian chạy script: 151 giây (exit 0). Log đầy đủ: `R/runs/tfidf-calib/out.log`.

## Độ lệch ước lượng
- Độ lệch = 0.91819 − OOF group = 0.91819 − 0.9168 = **0.0014** (thô).
- Lưu ý: con số 0.91819 là điểm của bài người bạn (không rõ cấu hình). Cấu hình bài đó có thể khác (đặc trưng, mô hình, cách chia fold), nên độ lệch này chỉ mang tính tham khảo, không kết luận được gì về OOF của M1/M2.
- Chênh lệch giữa hai splitter ở đây rất nhỏ (0.0003), nên việc chia nhóm prefix-60 gần như không làm OOF bị thổi phồng trong cấu hình TF-IDF này.

## Ghi chú
- Không submit, không git, không sửa code.
- Thời gian mỗi splitter 1–1.5 phút, không cần giảm bớt cấu hình.
