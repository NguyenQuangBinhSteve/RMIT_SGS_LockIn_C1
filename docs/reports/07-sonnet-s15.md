# Báo cáo Sonnet – S15 (hiệu chuẩn TF-IDF cục bộ)

Script: `tools/tfidf_calib.py`. Tự viết, chỉ dùng sklearn/pandas đã có trong `.venv`. Chỉ để đo chênh lệch OOF cục bộ so với public 0.91819 của baseline TF-IDF, KHÔNG dùng để nộp. Không push, không submit, không git.

## Nội dung
- Đặc trưng: `TfidfVectorizer` char 2–5 và word 1–2 (min_df=2, sublinear_tf) ghép bằng FeatureUnion; `LogisticRegression(C=10, class_weight='balanced', max_iter=300)`. Vectorizer được fit lại trong từng fold (cross_val_predict + Pipeline) nên không rò rỉ.
- In macro F1 OOF cho `StratifiedGroupKFold` (K=3, seed 42, nhóm = text chữ thường, `\W+` thành khoảng trắng, strip, 60 ký tự đầu, giống notebook) và `StratifiedKFold` (K=3, seed 42), kèm số phút.
- Tham số: `--data` (mặc định `R/data`, tự tìm `train.csv` ở mọi cấp), `--n` (lấy mẫu n dòng để chạy thử).

## Cách chạy (Haiku)
```
R=/home/quocanh/quoc_anh/LockIn_RMIT_HACKATHON_2026
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
$R/.venv/bin/kaggle competitions download rmit-hackathon-2026 -p $R/data   # rồi giải nén vào $R/data (unzip)
$R/.venv/bin/python -I $R/tools/tfidf_calib.py
```

## Kiểm tra
Dữ liệu thật chưa tải. Đã chạy thử `.venv/bin/python -I tools/tfidf_calib.py --data <thư mục giả> --n 200` trên CSV giả (300 dòng) với kết quả in đúng định dạng, ~1.6 giây. Chưa biết thời gian trên 14000 dòng thật (char 2–5 nặng; ước vài phút mỗi splitter, có thể hơn). Nếu quá chậm, giảm bằng `max_features` cho vectorizer char.
