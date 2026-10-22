import gc
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup

from src.models import MultiSampleClassifier
from src.utils import log_memory


class FGM:
    """Fast Gradient Method (FGM) áp dụng nhiễu đối kháng lên tầng Embeddings."""
    def __init__(self, model, emb_name="word_embeddings", epsilon=1.0):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, fgm=None):
    model.train()
    total_loss = 0.0

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 1. Forward và backward chuẩn (Phase 1)
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

        # 2. Học đối kháng FGM (Phase 2)
        if fgm is not None:
            fgm.attack()
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs_adv = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
                loss_adv = criterion(outputs_adv[0], labels)

            scaler.scale(loss_adv).backward()
            fgm.restore()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        # Chỉ tăng bước scheduler nếu optimizer thực sự cập nhật bước đi
        if scale_before <= scaler.get_scale():
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
    """Huấn luyện và đánh giá trên 1 fold với MultiSample Dropout và 2-Phase FGM Curriculum."""
    print(f"\n{'=' * 20} FOLD {fold + 1}/{cfg.N_FOLDS} {'=' * 20}", flush=True)
    log_memory(f"Fold {fold + 1}: before model load")

    # Khởi tạo model với Custom Multi-Sample Dropout & Mean Pooling Head
    base_model = MultiSampleClassifier(
        model_name_or_path=cfg.MODEL_DIR,
        num_classes=3,
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

    # Khởi tạo bộ tấn công đối kháng FGM
    fgm = FGM(model, epsilon=getattr(cfg, "FGM_EPSILON", 1.0)) if getattr(cfg, "USE_FGM", True) else None
    adv_start = getattr(cfg, "ADV_START_EPOCH", 3)

    for epoch in range(cfg.EPOCHS):
        # Chiến thuật 2 Phase:
        # Phase 1: Train chuẩn để model nhanh chóng hội tụ
        # Phase 2: Kích hoạt FGM để làm phẳng loss landscape và tăng độ lì
        current_fgm = fgm if (fgm is not None and (epoch + 1) >= adv_start) else None
        phase_str = "Phase 2: FGM Adv" if current_fgm is not None else "Phase 1: Standard"

        loss = train_epoch(
            model, train_loader, optimizer,
            scheduler, criterion, scaler, cfg.DEVICE, fgm=current_fgm
        )
        probs = eval_model(model, val_loader, cfg.DEVICE)
        preds = probs.argmax(axis=1)

        macro_f1 = f1_score(
            y_val_true, preds, average="macro", zero_division=0
        )
        acc = accuracy_score(y_val_true, preds)

        print(
            f"Epoch {epoch + 1}/{cfg.EPOCHS} [{phase_str}] | "
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
