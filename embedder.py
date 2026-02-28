"""Compatibility entrypoint for the keystroke embedder training stack."""

from embedder_core.cli import main
from embedder_core.constants import MASK, PAD, SPECIAL_TOKENS, UNK
from embedder_core.data import AugmentConfig, KeystrokeUserDataset, collate_fn
from embedder_core.model import KeystrokeTransformer
from embedder_core.tokens import KeypressToken, parse_run_to_tokens
from embedder_core.train import TrainConfig, evaluate, load_model_weights, train
from embedder_core.utils import accuracy_from_logits, cosine_with_warmup, set_seed
from embedder_core.visualize import save_pca_embedding_plot
from embedder_core.vocab import build_keysym_vocab, normalize_keysym

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


if __name__ == "__main__":
    raise SystemExit(main())
