import gc
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)

from src.config import ID2LABEL, LABEL2ID
from src.utils import log_memory


def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    total_loss = 0.0

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
            logits = outputs[0]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.detach().item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_probs = []

    for batch in loader:
        input_ids = batch[0].to(device, non_blocking=True)
        attention_mask = batch[1].to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
            probs = torch.softmax(outputs[0].float(), dim=1)

        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def train_one_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_dataset,
    test_loader: DataLoader,
    train_collate,
    criterion: nn.Module,
    n_gpus: int,
    cfg,
    y_val_true: np.ndarray,
):
    """Huấn luyện và đánh giá trên 1 fold, trả về best_val_probs và fold_test_probs."""
    print(f"\n{'=' * 20} FOLD {fold + 1}/{cfg.N_FOLDS} {'=' * 20}", flush=True)
    log_memory(f"Fold {fold + 1}: before model load")

    # Khởi tạo model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL_DIR,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    base_model.to(cfg.DEVICE)

    if n_gpus > 1:
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    del base_model
    gc.collect()
    log_memory(f"Fold {fold + 1}: after model load")

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        Subset(train_dataset, val_idx),
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * cfg.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(total_steps * cfg.WARMUP_RATIO),
        total_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(cfg.DEVICE.type == "cuda"),
    )

    best_score = -1.0
    best_val_probs = None
    checkpoint = os.path.join(cfg.OUTPUT_DIR, f"{cfg.EXP_NAME}_fold_{fold}.pt")

    for epoch in range(cfg.EPOCHS):
        loss = train_epoch(
            model, train_loader, optimizer,
            scheduler, criterion, scaler, cfg.DEVICE,
        )
        probs = eval_model(model, val_loader, cfg.DEVICE)
        preds = probs.argmax(axis=1)

        macro_f1 = f1_score(
            y_val_true, preds, average="macro", zero_division=0
        )
        acc = accuracy_score(y_val_true, preds)

        print(
            f"Epoch {epoch + 1}/{cfg.EPOCHS} | "
            f"Loss={loss:.4f} | Acc={acc:.4f} | "
            f"Macro F1={macro_f1:.4f}",
            flush=True,
        )

        if macro_f1 > best_score:
            best_score = macro_f1
            best_val_probs = probs.copy()

            state = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(state, checkpoint)

        del probs, preds
        log_memory(f"Fold {fold + 1}, epoch {epoch + 1}")

    if best_val_probs is None:
        raise RuntimeError(f"Fold {fold + 1} không có validation predictions")

    # Load best weights vào chính model hiện tại, không tạo model thứ hai
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    target_model = (
        model.module
        if isinstance(model, nn.DataParallel)
        else model
    )
    target_model.load_state_dict(state)
    del state

    fold_test_probs = eval_model(model, test_loader, cfg.DEVICE)

    del model, optimizer, scheduler, scaler
    del train_loader, val_loader
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log_memory(f"Fold {fold + 1}: cleanup complete")

    return best_val_probs, fold_test_probs

