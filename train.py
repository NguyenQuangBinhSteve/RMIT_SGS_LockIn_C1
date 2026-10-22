import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

from src.config import ID2LABEL, LABEL2ID, Config, get_config, parse_args
from src.dataset import (
    TextDataset,
    create_test_loader,
    load_and_validate_data,
    make_collator,
)
from src.engine import train_one_fold
from src.utils import (
    calculate_class_weights,
    log_memory,
    print_system_info,
    save_results,
    seed_everything,
    summarize_oof,
)


def main(cfg: Config = None):
    # Nếu không truyền cfg vào (ví dụ chạy từ CLI), tự động parse command-line arguments
    if cfg is None:
        cfg = parse_args()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    seed_everything(cfg.SEED)
    n_gpus = print_system_info(cfg)

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT  : {cfg.EXP_NAME}")
    print(f"MODEL PATH  : {cfg.MODEL_DIR}")
    print(f"BATCH SIZE  : {cfg.BATCH_SIZE}")
    print(f"EPOCHS      : {cfg.EPOCHS}")
    print(f"USE FGM     : {cfg.USE_FGM} (Phase 2 starts at Epoch {cfg.ADV_START_EPOCH})")
    print(f"OUTPUT DIR  : {cfg.OUTPUT_DIR}")
    print(f"{'=' * 60}\n", flush=True)


    # 1. Load và kiểm định dữ liệu
    log_memory("Before CSV")
    train_df, test_df, counts = load_and_validate_data(
        data_dir=cfg.DATA_DIR,
        label2id=LABEL2ID,
        n_folds=cfg.N_FOLDS,
    )
    log_memory("After CSV")

    # 2. Chuẩn bị Tokenizer, Datasets và DataLoaders
    print(f"\nLoading tokenizer for {cfg.MODEL_DIR}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR)

    train_dataset = TextDataset(
        train_df["text"],
        tokenizer,
        cfg.MAX_LEN,
        labels=train_df["target"].to_numpy(),
    )
    test_dataset = TextDataset(
        test_df["text"],
        tokenizer,
        cfg.MAX_LEN,
    )

    train_collate = make_collator(tokenizer, has_labels=True)
    test_collate = make_collator(tokenizer, has_labels=False)

    test_loader = create_test_loader(
        test_dataset,
        collate_fn=test_collate,
        batch_size=cfg.BATCH_SIZE * 2,
        num_workers=cfg.NUM_WORKERS,
    )

    # 3. Thiết lập hàm Loss (có tính class weights bù trừ mất cân bằng nhãn)
    criterion = calculate_class_weights(
        counts=counts,
        n_samples=len(train_df),
        device=cfg.DEVICE,
    )

    # 4. Stratified K-Fold Training Loop
    skf = StratifiedKFold(
        n_splits=cfg.N_FOLDS,
        shuffle=True,
        random_state=cfg.SEED,
    )

    oof_probs = np.zeros((len(train_df), 3), dtype=np.float32)
    test_probs = np.zeros((len(test_df), 3), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["target"])
    ):
        y_val_true = train_df.iloc[val_idx]["target"].to_numpy()

        fold_val_probs, fold_test_probs = train_one_fold(
            fold=fold,
            train_idx=train_idx,
            val_idx=val_idx,
            train_dataset=train_dataset,
            test_loader=test_loader,
            train_collate=train_collate,
            criterion=criterion,
            n_gpus=n_gpus,
            cfg=cfg,
            y_val_true=y_val_true,
        )

        oof_probs[val_idx] = fold_val_probs
        test_probs += fold_test_probs / cfg.N_FOLDS

    # 5. Đánh giá OOF và xuất kết quả
    summarize_oof(train_df["target"], oof_probs, ID2LABEL)
    save_results(
        output_dir=cfg.OUTPUT_DIR,
        exp_name=cfg.EXP_NAME,
        oof_probs=oof_probs,
        test_probs=test_probs,
        test_df=test_df,
        id2label=ID2LABEL,
    )


if __name__ == "__main__":
    main()
