from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .capture import normalize_raw_run
from .config import DEFAULT_EMBEDDER_CHECKPOINT_PATH, DEFAULT_EMBEDDER_FINAL_WEIGHTS_PATH


@dataclass
class EmbedderRuntime:
    model: Any
    stoi: Dict[str, int]
    device: str
    max_tokens: int
    clip_ms: float
    log1p: bool
    weights_path: Path
    checkpoint_path: Optional[Path]

    @property
    def feature_dim(self) -> int:
        return int(self.model.d_model)

    def build_feature_vector_from_raw_run(self, raw_run: Dict[str, Any]) -> np.ndarray:
        import torch
        from embedder_core.tokens import parse_run_to_tokens

        normalized = normalize_raw_run(raw_run)
        tokens = parse_run_to_tokens(normalized["events"], self.stoi)
        if not tokens:
            raise ValueError("Raw run has no usable keypress tokens for embedder inference.")

        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens]

        keysym_ids = torch.tensor([[t.keysym_id for t in tokens]], dtype=torch.long, device=self.device)
        dwell = torch.tensor([[t.dwell_ms for t in tokens]], dtype=torch.float32, device=self.device)
        flight = torch.tensor([[t.flight_ms for t in tokens]], dtype=torch.float32, device=self.device)
        attn_mask = torch.ones((1, keysym_ids.shape[1]), dtype=torch.bool, device=self.device)

        dwell = torch.clamp(dwell, 0.0, float(self.clip_ms))
        flight = torch.clamp(flight, 0.0, float(self.clip_ms))
        if self.log1p:
            dwell = torch.log1p(dwell)
            flight = torch.log1p(flight)

        with torch.no_grad():
            emb = self.model.encode(keysym_ids, dwell, flight, attn_mask)
        return emb.detach().cpu().numpy().reshape(-1).astype(np.float32)


def _extract_model_state(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("model_state"), dict):
        return payload["model_state"]
    if isinstance(payload, dict):
        if payload and all(hasattr(v, "shape") for v in payload.values()):
            return payload
    raise ValueError("Invalid embedder weights file format.")


def _infer_arch(state: Dict[str, Any], train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    key_emb = state.get("key_emb.weight")
    head_w = state.get("head.1.weight")
    if key_emb is None or head_w is None:
        raise ValueError("Weights are missing required tensors (key_emb/head).")

    vocab_size = int(key_emb.shape[0])
    d_model = int(key_emb.shape[1])
    num_classes = int(head_w.shape[0])

    layer_ids = set()
    for name in state.keys():
        if name.startswith("encoder.layers."):
            parts = name.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_ids.add(int(parts[2]))
    num_layers = max(layer_ids) + 1 if layer_ids else int(train_cfg.get("num_layers", 3))

    ff_w = state.get("encoder.layers.0.linear1.weight")
    dim_ff = int(ff_w.shape[0]) if ff_w is not None else int(train_cfg.get("dim_ff", max(2 * d_model, 128)))

    nhead = int(train_cfg.get("nhead", 4))
    dropout = float(train_cfg.get("dropout", 0.1))
    use_cls_token = "cls" in state

    return {
        "num_classes": num_classes,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_ff": dim_ff,
        "dropout": dropout,
        "use_cls_token": use_cls_token,
    }


def _load_torch_payload(path: Path) -> Any:
    import torch

    return torch.load(str(path), map_location="cpu")


def load_embedder_runtime(
    weights_path: Path,
    checkpoint_path: Optional[Path] = None,
) -> EmbedderRuntime:
    import torch
    from embedder_core.model import KeystrokeTransformer

    if not weights_path.exists():
        raise FileNotFoundError(f"Embedder weights file not found: {weights_path}")

    weights_payload = _load_torch_payload(weights_path)
    model_state = _extract_model_state(weights_payload)

    meta_payload = weights_payload if isinstance(weights_payload, dict) else {}
    stoi = meta_payload.get("stoi") if isinstance(meta_payload, dict) else None
    train_cfg = meta_payload.get("train_cfg", {}) if isinstance(meta_payload, dict) else {}
    aug_cfg = meta_payload.get("aug_cfg", {}) if isinstance(meta_payload, dict) else {}

    ckpt_used = None
    if not isinstance(stoi, dict):
        if checkpoint_path is None:
            raise ValueError("Embedder weights file has no vocab metadata and no checkpoint path was provided.")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Embedder weights file lacks vocab metadata and checkpoint was not found: {checkpoint_path}"
            )
        ckpt_payload = _load_torch_payload(checkpoint_path)
        if not isinstance(ckpt_payload, dict) or not isinstance(ckpt_payload.get("stoi"), dict):
            raise ValueError("Checkpoint file must include 'stoi' to run embedder inference.")
        stoi = ckpt_payload["stoi"]
        if not train_cfg:
            train_cfg = ckpt_payload.get("train_cfg", {})
        if not aug_cfg:
            aug_cfg = ckpt_payload.get("aug_cfg", {})
        ckpt_used = checkpoint_path

    arch = _infer_arch(model_state, train_cfg if isinstance(train_cfg, dict) else {})
    model = KeystrokeTransformer(**arch)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    device = "cpu"
    model = model.to(device)

    max_tokens = int((train_cfg or {}).get("max_tokens", 256))
    clip_ms = float((aug_cfg or {}).get("clip_ms", 800.0))
    log1p = bool((aug_cfg or {}).get("log1p", True))

    return EmbedderRuntime(
        model=model,
        stoi=stoi,
        device=device,
        max_tokens=max_tokens,
        clip_ms=clip_ms,
        log1p=log1p,
        weights_path=weights_path,
        checkpoint_path=ckpt_used,
    )


def try_load_default_embedder_runtime() -> Tuple[Optional[EmbedderRuntime], str]:
    try:
        runtime = load_embedder_runtime(
            weights_path=DEFAULT_EMBEDDER_FINAL_WEIGHTS_PATH,
            checkpoint_path=DEFAULT_EMBEDDER_CHECKPOINT_PATH,
        )
    except Exception as ex:
        return None, f"Embedder disabled: {ex}"

    if runtime.checkpoint_path is None:
        msg = f"Embedder enabled from {runtime.weights_path.name}"
    else:
        msg = (
            f"Embedder enabled from {runtime.weights_path.name} "
            f"(vocab metadata from {runtime.checkpoint_path.name})"
        )
    return runtime, msg
