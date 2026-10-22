import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, labels=None):
        self.texts = pd.Series(texts).fillna("").astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def make_collator(tokenizer, has_labels: bool):
    padder = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        return_tensors="pt",
    )

    def collate(batch):
        labels = None
        if has_labels:
            labels = torch.tensor(
                [item.pop("labels") for item in batch],
                dtype=torch.long,
            )

        features = padder(batch)
        if has_labels:
            return (
                features["input_ids"],
                features["attention_mask"],
                labels,
            )
        return features["input_ids"], features["attention_mask"]

    return collate


def load_and_validate_data(data_dir: str, label2id: dict, n_folds: int):
    """Đọc và kiểm tra tính toàn vẹn của dữ liệu train / test."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Không tìm thấy DATA_DIR: {data_dir}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    required_train = {"text", "label"}
    required_test = {"id", "text"}

    if not required_train.issubset(train_df.columns):
        raise ValueError(f"train.csv cần các cột: {required_train}")
    if not required_test.issubset(test_df.columns):
        raise ValueError(f"test.csv cần các cột: {required_test}")

    train_df["target"] = train_df["label"].map(label2id)

    if train_df["target"].isna().any():
        bad_labels = train_df.loc[
            train_df["target"].isna(), "label"
        ].unique()
        raise ValueError(f"Label không hợp lệ: {bad_labels}")

    train_df["target"] = train_df["target"].astype(int)

    counts = np.bincount(
        train_df["target"].to_numpy(),
        minlength=3,
    )
    if np.any(counts == 0):
        raise ValueError(f"Có class không xuất hiện trong train: {counts}")
    if counts.min() < n_folds:
        raise ValueError(
            f"Mỗi class cần ít nhất {n_folds} mẫu để StratifiedKFold. "
            f"Counts={counts}"
        )

    return train_df, test_df, counts


def create_test_loader(test_dataset: Dataset, collate_fn, batch_size: int, num_workers: int = 0) -> DataLoader:
    """Tạo DataLoader cho tập test."""
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

