"""Standalone embedder package."""

from .cli import main
from .constants import MASK, PAD, SPECIAL_TOKENS, UNK
from .data import AugmentConfig, KeystrokeUserDataset, collate_fn
from .model import KeystrokeTransformer
from .tokens import KeypressToken, parse_run_to_tokens
from .train import TrainConfig, evaluate, load_model_weights, train
from .utils import accuracy_from_logits, cosine_with_warmup, set_seed
from .visualize import save_pca_embedding_plot
from .vocab import build_keysym_vocab, normalize_keysym

__all__ = [
    "PAD",
    "UNK",
    "MASK",
    "SPECIAL_TOKENS",
    "KeypressToken",
    "AugmentConfig",
    "TrainConfig",
    "KeystrokeUserDataset",
    "KeystrokeTransformer",
    "accuracy_from_logits",
    "build_keysym_vocab",
    "collate_fn",
    "cosine_with_warmup",
    "evaluate",
    "load_model_weights",
    "main",
    "normalize_keysym",
    "parse_run_to_tokens",
    "save_pca_embedding_plot",
    "set_seed",
    "train",
]
