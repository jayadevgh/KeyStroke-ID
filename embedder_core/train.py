from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .constants import MASK, PAD, UNK
from .data import AugmentConfig, KeystrokeUserDataset, collate_fn
from .model import KeystrokeTransformer
from .utils import accuracy_from_logits, cosine_with_warmup, set_seed
from .visualize import save_pca_embedding_plot
from .vocab import build_keysym_vocab


@dataclass
class TrainConfig:
    epochs: int = 0
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    val_frac: float = 0.15
    max_tokens: int = 256
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    early_stop_patience: int = 8
    save_path: str = "keystroke_user_classifier_keysym.pt"
    final_weights_path: str = "keystroke_user_classifier_keysym_final_weights.pt"
    pca_plot_path: str = "keystroke_user_classifier_keysym_pca.png"


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    for batch in loader:
        for key in ("keysym_ids", "dwell", "flight", "attn_mask", "y"):
            batch[key] = batch[key].to(device)
        logits = model(batch["keysym_ids"], batch["dwell"], batch["flight"], batch["attn_mask"])
        loss = criterion(logits, batch["y"])
        losses.append(loss.item())
        accuracies.append(accuracy_from_logits(logits, batch["y"]))
    return float(np.mean(losses)), float(np.mean(accuracies))


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and isinstance(payload.get("model_state"), dict):
        return payload["model_state"]
    if isinstance(payload, dict) and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload
    raise ValueError("Weights file must be a torch state_dict or checkpoint with a 'model_state' key.")


def load_model_weights(model: nn.Module, path: str, device: str) -> None:
    payload = torch.load(path, map_location=device)
    state_dict = _extract_state_dict(payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if not missing and not unexpected:
        print(f"Loaded weights from {path}")
    else:
        print(
            f"Loaded weights from {path} with partial match "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )


def train(
    user_json_paths: List[str],
    min_freq: int = 2,
    *,
    load_weights_path: Optional[str] = None,
    save_path: Optional[str] = None,
    final_weights_path: Optional[str] = None,
    pca_plot_path: Optional[str] = None,
) -> None:
    cfg = TrainConfig()
    if save_path is not None:
        cfg.save_path = save_path
    if final_weights_path is not None:
        cfg.final_weights_path = final_weights_path
    if pca_plot_path is not None:
        cfg.pca_plot_path = pca_plot_path

    aug = AugmentConfig()
    set_seed(cfg.seed)

    stoi, itos = build_keysym_vocab(user_json_paths, min_freq=min_freq)
    pad_id = stoi[PAD]
    unk_id = stoi[UNK]
    mask_id = stoi[MASK]

    train_ds = KeystrokeUserDataset(
        user_json_paths=user_json_paths,
        stoi=stoi,
        split="train",
        val_frac=cfg.val_frac,
        seed=cfg.seed,
        max_tokens=cfg.max_tokens,
    )
    val_ds = KeystrokeUserDataset(
        user_json_paths=user_json_paths,
        stoi=stoi,
        split="val",
        val_frac=cfg.val_frac,
        seed=cfg.seed,
        max_tokens=cfg.max_tokens,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda batch: collate_fn(batch, aug, True, pad_id, mask_id, unk_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda batch: collate_fn(batch, aug, False, pad_id, mask_id, unk_id),
    )

    model = KeystrokeTransformer(
        num_classes=len(user_json_paths),
        vocab_size=len(itos),
        d_model=160,
        nhead=4,
        num_layers=3,
        dim_ff=320,
        dropout=0.1,
        use_cls_token=True,
    ).to(cfg.device)
    if load_weights_path:
        load_model_weights(model, path=load_weights_path, device=cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_steps = cfg.epochs * max(1, len(train_loader))
    step = 0

    best_val_acc = -1.0
    patience = 0
    best_model_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_accuracies = []

        for batch in train_loader:
            for key in ("keysym_ids", "dwell", "flight", "attn_mask", "y"):
                batch[key] = batch[key].to(cfg.device)

            lr_scale = cosine_with_warmup(step, total_steps, cfg.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * lr_scale

            logits = model(batch["keysym_ids"], batch["dwell"], batch["flight"], batch["attn_mask"])
            loss = criterion(logits, batch["y"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy_from_logits(logits, batch["y"]))
            step += 1

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(np.mean(epoch_accuracies))
        val_loss, val_acc = evaluate(model, val_loader, cfg.device)

        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            patience = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "stoi": stoi,
                    "itos": itos,
                    "train_cfg": cfg.__dict__,
                    "aug_cfg": aug.__dict__,
                    "num_classes": len(user_json_paths),
                },
                cfg.save_path,
            )
            print(f"  saved best -> {cfg.save_path} (val_acc={best_val_acc:.3f})")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("Early stopping.")
                break

    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "train_cfg": cfg.__dict__,
            "aug_cfg": aug.__dict__,
            "num_classes": len(user_json_paths),
        },
        cfg.final_weights_path,
    )
    print(f"Saved final weights -> {cfg.final_weights_path}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    save_pca_embedding_plot(
        model=model,
        user_json_paths=user_json_paths,
        stoi=stoi,
        cfg=cfg,
        pad_id=pad_id,
        mask_id=mask_id,
        unk_id=unk_id,
        output_path=cfg.pca_plot_path,
    )
