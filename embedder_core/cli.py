import argparse
import sys
from typing import List, Optional

from .train import train


def main(argv: Optional[List[str]] = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(description="Train the keystroke embedder.")
    parser.add_argument("json_paths", nargs="+", help="User JSON files (one file per class/user).")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum token frequency for vocab inclusion.")
    parser.add_argument(
        "--load-weights",
        default=None,
        help="Optional path to an existing weights/checkpoint file to initialize the model from.",
    )
    parser.add_argument("--save-path", default=None, help="Optional path for best-checkpoint output.")
    parser.add_argument("--final-weights-path", default=None, help="Optional path for final weights output.")
    parser.add_argument("--pca-plot-path", default=None, help="Optional path for PCA plot output.")
    args = parser.parse_args(args_in)

    paths = list(args.json_paths)
    if len(paths) < 2:
        print(
            "Pass >=2 user json paths (one per person). "
            "Example: python embedder.py u0.json u1.json u2.json u3.json"
        )
        return 1

    train(
        paths,
        min_freq=args.min_freq,
        load_weights_path=args.load_weights,
        save_path=args.save_path,
        final_weights_path=args.final_weights_path,
        pca_plot_path=args.pca_plot_path,
    )
    return 0
