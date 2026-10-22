"""
Script ghép mô hình (Ensemble Blending):
Kết hợp xác suất của Deep Learning (DistilBERT) và Machine Learning (TF-IDF + Ridge/LR).
Chạy đoạn này ngay sau khi DistilBERT hoàn thành huấn luyện.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Đường dẫn file
TEST_PATH = "/kaggle/input/competitions/rmit-hackathon-2026/test.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "data/raw/test.csv"

ID2LABEL = {0: "BENIGN", 1: "JAILBREAK", 2: "PROMPT_INJECTION"}

def blend_predictions(
    distilbert_test_npy="distilbert_test.npy",
    tfidf_test_npy="test_blend.npy",
    weight_dl=0.65,
    w_prompt_injection=1.1,
    output_csv="submission_final_ensemble.csv"
):
    test_df = pd.read_csv(TEST_PATH)

    # 1. Load xác suất nếu có file npy
    if os.path.exists(distilbert_test_npy) and os.path.exists(tfidf_test_npy):
        p_dl = np.load(distilbert_test_npy)
        p_ml = np.load(tfidf_test_npy)
        probs = weight_dl * p_dl + (1 - weight_dl) * p_ml
    else:
        print("Chưa có đủ file .npy, kiểm tra file submission csv có sẵn...")
        # Fallback: Nếu chỉ có file submission
        return

    # 2. Hiệu chuẩn trọng số lớp PROMPT_INJECTION (ngăn chặn bias do lớp thiểu số)
    probs[:, 2] *= w_prompt_injection

    # 3. Rule-based override: Bắt lỗi chèn tiếng Việt trong prompt ngoại ngữ (Độ chuẩn xác 97.8% từ EDA)
    vi_pattern = r'[đĐưƯơƠạảấầẩẫậắằẳẵặẹẻẽếềểễệỉĩịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ]'
    has_vi = test_df["text"].fillna("").astype(str).str.contains(vi_pattern, regex=True).values
    probs[has_vi, :] = [0.0, 0.0, 1.0]

    # 4. Xuất kết quả
    preds = np.argmax(probs, axis=1)
    sub = pd.DataFrame({
        "id": test_df["id"],
        "label": [ID2LABEL[p] for p in preds]
    })
    sub.to_csv(output_csv, index=False)
    print(f" Đã tạo file {output_csv}!")
    print("Phân bố nhãn dự đoán cuối cùng:")
    print(sub["label"].value_counts(normalize=True).round(4) * 100)

if __name__ == "__main__":
    blend_predictions()
