import os
import random
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_memory(tag: str):
    ram = psutil.virtual_memory()
    print(
        f"[{tag}] RAM available: "
        f"{ram.available / 1024**3:.2f} GB",
        flush=True,
    )
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(
                f"  GPU {i}: allocated="
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB, "
                f"reserved="
                f"{torch.cuda.memory_reserved(i) / 1024**3:.2f} GB",
                flush=True,
            )


def print_system_info(cfg) -> int:
    """In thông số môi trường phần cứng và trả về số lượng GPUs."""
    n_gpus = torch.cuda.device_count()
    print("=" * 60)
    print("PyTorch:", torch.__version__)
    print("Device:", cfg.DEVICE)
    print("GPU count:", n_gpus)

    for i in range(n_gpus):
        print(i, torch.cuda.get_device_name(i))

    print("Batch size:", cfg.BATCH_SIZE)
    print("=" * 60, flush=True)

    if not torch.cuda.is_available():
        print("WARNING: CUDA không khả dụng, đang chạy CPU.")

    return n_gpus


def calculate_class_weights(counts: np.ndarray, n_samples: int, device: torch.device) -> nn.CrossEntropyLoss:
    """Tính class weights bù trừ mất cân bằng nhãn và trả về criterion."""
    weights = torch.tensor(
        n_samples / (3.0 * counts),
        dtype=torch.float32,
        device=device,
    )
    return nn.CrossEntropyLoss(weight=weights)


def summarize_oof(y_true, oof_probs: np.ndarray, id2label: dict) -> float:
    """Tính và in báo cáo chi tiết kết quả Out-Of-Fold (OOF)."""
    oof_preds = oof_probs.argmax(axis=1)
    overall_f1 = f1_score(
        y_true, oof_preds,
        average="macro", zero_division=0,
    )

    print("\n" + "=" * 60)
    print(f"5-FOLD OOF MACRO F1: {overall_f1:.4f}")
    print("=" * 60)
    print(
        classification_report(
            y_true,
            oof_preds,
            labels=[0, 1, 2],
            target_names=[id2label[i] for i in range(3)],
            zero_division=0,
        )
    )
    return overall_f1


def save_results(
    output_dir: str,
    exp_name: str,
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    test_df: pd.DataFrame,
    id2label: dict,
):
    """Lưu OOF, test probabilities và file submission csv theo exp_name linh hoạt."""
    oof_save_path = os.path.join(output_dir, f"{exp_name}_oof.npy")
    test_save_path = os.path.join(output_dir, f"{exp_name}_test.npy")
    np.save(oof_save_path, oof_probs)
    np.save(test_save_path, test_probs)

    final_preds = test_probs.argmax(axis=1)
    submission = pd.DataFrame({
        "id": test_df["id"],
        "label": [id2label[int(p)] for p in final_preds],
    })

    submission_path = os.path.join(output_dir, f"submission_{exp_name}.csv")
    submission.to_csv(submission_path, index=False)

    print("Saved OOF:", oof_save_path)
    print("Saved Test Probs:", test_save_path)
    print("Saved Submission:", submission_path)
    print(submission["label"].value_counts())
    print("Done!", flush=True)


