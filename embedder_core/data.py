import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .tokens import KeypressToken, parse_run_to_tokens


class KeystrokeUserDataset(Dataset):
    """
    Expects one JSON per user:
      { "enrollment_runs": [ {"events": [...]}, ... ] }
    """

    def __init__(
        self,
        user_json_paths: List[str],
        stoi: Dict[str, int],
        split: str = "train",
        val_frac: float = 0.20,
        seed: int = 42,
        max_tokens: int = 256,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "all")
        self.split = split
        self.max_tokens = max_tokens
        self.stoi = stoi

        all_samples: List[Tuple[List[KeypressToken], int]] = []
        for label, path in enumerate(user_json_paths):
            with open(path, "r", encoding="utf-8") as handle:
                blob = json.load(handle)
            runs = blob.get("enrollment_runs", [])
            for run in runs:
                tokens = parse_run_to_tokens(run.get("events", []), stoi)
                if len(tokens) < 5:
                    continue
                all_samples.append((tokens, label))

        rng = random.Random(seed)
        by_label: Dict[int, List[Tuple[List[KeypressToken], int]]] = {}
        for sample in all_samples:
            by_label.setdefault(sample[1], []).append(sample)

        train_samples: List[Tuple[List[KeypressToken], int]] = []
        val_samples: List[Tuple[List[KeypressToken], int]] = []
        for samples in by_label.values():
            rng.shuffle(samples)
            n_val = max(1, int(len(samples) * val_frac))
            val_samples.extend(samples[:n_val])
            train_samples.extend(samples[n_val:])

        if split == "train":
            self.samples = train_samples
        elif split == "val":
            self.samples = val_samples
        else:
            self.samples = all_samples

        rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tokens, label = self.samples[idx]
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens]

        keysym_ids = torch.tensor([t.keysym_id for t in tokens], dtype=torch.long)
        dwell = torch.tensor([t.dwell_ms for t in tokens], dtype=torch.float32)
        flight = torch.tensor([t.flight_ms for t in tokens], dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return {"keysym_ids": keysym_ids, "dwell": dwell, "flight": flight, "y": y}


@dataclass
class AugmentConfig:
    enable: bool = True
    speed_scale_min: float = 0.9
    speed_scale_max: float = 1.1
    jitter_std_ms: float = 10.0
    token_drop_prob: float = 0.06
    key_mask_prob: float = 0.15
    key_unk_prob: float = 0.03
    clip_ms: float = 800.0
    log1p: bool = True


def collate_fn(
    batch: List[Dict[str, Any]],
    aug: AugmentConfig,
    is_train: bool,
    pad_id: int,
    mask_id: int,
    unk_id: int,
) -> Dict[str, torch.Tensor]:
    ks_seqs: List[torch.Tensor] = []
    dwell_seqs: List[torch.Tensor] = []
    flight_seqs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []

    for ex in batch:
        ks = ex["keysym_ids"].clone()
        dwell = ex["dwell"].clone()
        flight = ex["flight"].clone()
        y = ex["y"]

        if is_train and aug.enable:
            if len(ks) > 8 and aug.token_drop_prob > 0:
                keep = torch.rand(len(ks)) > aug.token_drop_prob
                if keep.sum().item() >= 5:
                    ks = ks[keep]
                    dwell = dwell[keep]
                    flight = flight[keep]

            if aug.key_mask_prob > 0:
                mask = torch.rand(len(ks)) < aug.key_mask_prob
                ks[mask] = mask_id
            if aug.key_unk_prob > 0:
                mask = torch.rand(len(ks)) < aug.key_unk_prob
                ks[mask] = unk_id

            scale = random.uniform(aug.speed_scale_min, aug.speed_scale_max)
            dwell = dwell * scale
            flight = flight * scale

            if aug.jitter_std_ms > 0:
                dwell = dwell + torch.randn_like(dwell) * aug.jitter_std_ms
                flight = flight + torch.randn_like(flight) * aug.jitter_std_ms

        dwell = torch.clamp(dwell, 0.0, aug.clip_ms)
        flight = torch.clamp(flight, 0.0, aug.clip_ms)

        if aug.log1p:
            dwell = torch.log1p(dwell)
            flight = torch.log1p(flight)

        ks_seqs.append(ks)
        dwell_seqs.append(dwell)
        flight_seqs.append(flight)
        ys.append(y)

    lengths = torch.tensor([len(seq) for seq in ks_seqs], dtype=torch.long)
    max_len = int(lengths.max().item())

    ks_pad = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    dwell_pad = torch.zeros((len(batch), max_len), dtype=torch.float32)
    flight_pad = torch.zeros((len(batch), max_len), dtype=torch.float32)
    attn_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, (ks, dwell, flight) in enumerate(zip(ks_seqs, dwell_seqs, flight_seqs)):
        seq_len = len(ks)
        ks_pad[i, :seq_len] = ks
        dwell_pad[i, :seq_len] = dwell
        flight_pad[i, :seq_len] = flight
        attn_mask[i, :seq_len] = True

    y = torch.stack(ys)
    return {
        "keysym_ids": ks_pad,
        "dwell": dwell_pad,
        "flight": flight_pad,
        "attn_mask": attn_mask,
        "y": y,
        "lengths": lengths,
    }
