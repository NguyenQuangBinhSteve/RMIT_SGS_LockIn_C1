import argparse
import glob
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import classification_report, f1_score

from src.config import ID2LABEL, LABEL2ID, resolve_data_dir, resolve_output_dir


# Pattern phát hiện ký tự tiếng Việt
VI_PATTERN = r"[đĐưƯơƠạảấầẩẫậắằẳẵặẹẻẽếềểễệỉĩịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ]"


def optimize_class_weights(y_true: np.ndarray, oof_probs: np.ndarray) -> np.ndarray:
    """
    Dùng thuật toán Nelder-Mead tối ưu hóa bộ trọng số [w0, w1, w2]
    để trực tiếp TỐI ĐA HÓA điểm MACRO F1 trên tập Out-Of-Fold (OOF).
    """
    baseline_preds = np.argmax(oof_probs, axis=1)
    baseline_f1 = f1_score(y_true, baseline_preds, average="macro", zero_division=0)
    print(f"[*] Baseline OOF Macro F1 (argmax mặc định): {baseline_f1:.4f}")

    def loss_func(weights):
        # Tránh trọng số âm
        w = np.clip(weights, 0.1, 5.0)
        scaled_probs = oof_probs * w
        preds = np.argmax(scaled_probs, axis=1)
        # scipy minimize -> cần cực tiểu hóa âm F1
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        return -score

    init_weights = np.array([1.0, 1.0, 1.0])
    res = minimize(
        loss_func,
        init_weights,
        method="Nelder-Mead",
        options={"maxiter": 500, "disp": False},
    )

    best_w = np.clip(res.x, 0.1, 5.0)
    best_w = best_w / best_w[0]  # Chuẩn hóa theo lớp 0

    best_preds = np.argmax(oof_probs * best_w, axis=1)
    tuned_f1 = f1_score(y_true, best_preds, average="macro", zero_division=0)

    print(
        f"[+] Tuned OOF Macro F1 (sau khi chỉnh ngưỡng): {tuned_f1:.4f} (+{(tuned_f1 - baseline_f1)*100:.2f}%)"
    )
    print(
        f"[+] Bộ trọng số lớp tối ưu: BENIGN={best_w[0]:.2f}, JAILBREAK={best_w[1]:.2f}, INJECTION={best_w[2]:.2f}\n"
    )
    return best_w


def run_postprocess(
    exp_name: str = None,
    output_dir: str = None,
    data_dir: str = None,
    apply_vi_rule: bool = True,
):
    output_dir = output_dir or resolve_output_dir()
    data_dir = data_dir or resolve_data_dir()

    # 1. Tìm file xác suất
    test_npy_files = glob.glob(os.path.join(output_dir, "*_test.npy"))
    if not test_npy_files:
        raise FileNotFoundError(
            f"Không tìm thấy file *_test.npy nào trong {output_dir}"
        )

    if exp_name:
        test_npy_path = os.path.join(output_dir, f"{exp_name}_test.npy")
        oof_npy_path = os.path.join(output_dir, f"{exp_name}_oof.npy")
    else:
        # Lấy file test npy mới nhất
        test_npy_path = max(test_npy_files, key=os.path.getmtime)
        exp_name = os.path.basename(test_npy_path).replace("_test.npy", "")
        oof_npy_path = os.path.join(output_dir, f"{exp_name}_oof.npy")

    print("=" * 60)
    print(f"CHẠY HẬU XỬ LÝ (POST-PROCESSING) CHO: {exp_name}")
    print(f"Test Probs: {test_npy_path}")
    print(f"OOF Probs : {oof_npy_path}")
    print("=" * 60)

    # 2. Đọc Test CSV & Test Probs
    test_csv_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Không tìm thấy {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)
    test_probs = np.load(test_npy_path).copy()

    # 3. Bước 1: Tối ưu ngưỡng (Threshold Tuning) bằng tập OOF nếu có
    train_csv_path = os.path.join(data_dir, "train.csv")
    if os.path.exists(oof_npy_path) and os.path.exists(train_csv_path):
        print("\n[BƯỚC 1]: TỐI ƯU HÓA NGƯỠNG PHÂN LOẠI TRÊN TẬP OOF...")
        train_df = pd.read_csv(train_csv_path)
        y_true = train_df["label"].map(LABEL2ID).to_numpy()
        oof_probs = np.load(oof_npy_path)

        weights = optimize_class_weights(y_true, oof_probs)
        # Áp dụng bộ trọng số tối ưu lên xác suất tập test
        test_probs = test_probs * weights
    else:
        print(
            "\n[!] Không tìm thấy OOF, dùng trọng số mặc định [1.0, 1.0, 1.15] cho lớp thiểu số."
        )
        test_probs[:, 2] *= 1.15

    # 4. Bước 2: Áp dụng Luật Rule-based Tiếng Việt (EDA 97.8% Accuracy)
    if apply_vi_rule:
        print("[BƯỚC 2]: ÁP DỤNG LUẬT PHÁT HIỆN PROMPT INJECTION TIẾNG VIỆT...")
        has_vi = (
            test_df["text"]
            .fillna("")
            .astype(str)
            .str.contains(VI_PATTERN, regex=True)
            .to_numpy()
        )
        vi_count = has_vi.sum()
        print(
            f"[+] Tìm thấy {vi_count} câu có ký tự đặc trưng tiếng Việt trong tập test."
        )
        # Tăng mạnh xác suất PROMPT_INJECTION cho các câu tiếng Việt
        test_probs[has_vi, :] = [0.0, 0.0, 1.0]

    # 5. Bước 3: Xuất file submission hoàn chỉnh
    final_preds = np.argmax(test_probs, axis=1)
    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "label": [ID2LABEL[int(p)] for p in final_preds],
        }
    )

    sub_path = os.path.join(output_dir, f"submission_{exp_name}_postprocessed.csv")
    submission.to_csv(sub_path, index=False)

    print("\n" + "=" * 60)
    print(f" THÀNH CÔNG! ĐÃ TẠO FILE NỘP BÀI: {sub_path}")
    print("=" * 60)
    print("Phân bố nhãn dự đoán sau hậu xử lý:")
    print((submission["label"].value_counts(normalize=True) * 100).round(2).to_string())
    print("\nFile này đã sẵn sàng để nộp lên Leaderboard! ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hậu xử lý tối ưu hóa điểm Macro F1")
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Tên experiment cần hậu xử lý"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Đường dẫn thư mục output"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Đường dẫn thư mục data"
    )
    parser.add_argument(
        "--no_vi_rule", action="store_true", help="Không áp dụng luật tiếng Việt"
    )

    args = parser.parse_args()
    run_postprocess(
        exp_name=args.exp_name,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        apply_vi_rule=not args.no_vi_rule,
    )
