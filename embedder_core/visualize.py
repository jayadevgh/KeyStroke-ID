import os
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import AugmentConfig, KeystrokeUserDataset, collate_fn
from .model import KeystrokeTransformer


@torch.no_grad()
def save_pca_embedding_plot(
    model: KeystrokeTransformer,
    user_json_paths: List[str],
    stoi: Dict[str, int],
    cfg: Any,
    pad_id: int,
    mask_id: int,
    unk_id: int,
    output_path: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except Exception as exc:
        print(f"Skipping PCA plot (missing matplotlib/scikit-learn): {exc}")
        return

    aug = AugmentConfig(enable=False)
    dataset = KeystrokeUserDataset(
        user_json_paths=user_json_paths,
        stoi=stoi,
        split="all",
        val_frac=cfg.val_frac,
        seed=cfg.seed,
        max_tokens=cfg.max_tokens,
    )
    if len(dataset) == 0:
        print("Skipping PCA plot: no usable samples.")
        return

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda batch: collate_fn(batch, aug, False, pad_id, mask_id, unk_id),
    )

    model.eval()
    all_embeddings = []
    all_labels = []
    for batch in loader:
        for key in ("keysym_ids", "dwell", "flight", "attn_mask"):
            batch[key] = batch[key].to(cfg.device)
        emb = model.encode(batch["keysym_ids"], batch["dwell"], batch["flight"], batch["attn_mask"])
        all_embeddings.append(emb.detach().cpu().numpy())
        all_labels.append(batch["y"].detach().cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if embeddings.shape[0] < 2:
        print("Skipping PCA plot: need at least 2 embeddings.")
        return

    pca = PCA(n_components=2, random_state=cfg.seed)
    points = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_

    num_users = len(user_json_paths)
    if num_users <= 20:
        colors = plt.cm.tab20(np.linspace(0.0, 1.0, num_users))
    else:
        colors = plt.cm.nipy_spectral(np.linspace(0.0, 1.0, num_users))

    fig, ax = plt.subplots(figsize=(9, 7))
    for user_idx, json_path in enumerate(user_json_paths):
        mask = labels == user_idx
        if not np.any(mask):
            continue
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=24,
            alpha=0.8,
            color=colors[user_idx],
            label=os.path.basename(json_path),
            edgecolors="none",
        )

    ax.set_title("Keystroke Embeddings PCA")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved PCA embedding plot -> {output_path}")
