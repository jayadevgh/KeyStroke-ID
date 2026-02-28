import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from keystroke_app.storage import load_session_data
from keystroke_app.verifier import Verifier


def describe_distances(label: str, distances: np.ndarray):
    print(f"{label} distances:")
    print(f"  count = {distances.size}")
    print(f"  min   = {distances.min():.4f}")
    print(f"  max   = {distances.max():.4f}")
    print(f"  mean  = {distances.mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Suggest Mahalanobis distance thresholds from a dataset JSON.")
    parser.add_argument("dataset", type=Path, help="Path to keystroke_dataset.json")
    args = parser.parse_args()

    data = load_session_data(args.dataset)
    if len(data.enrollment_samples) < 3:
        raise SystemExit("Dataset needs at least 3 enrollment runs to tune a threshold.")

    verifier = Verifier()
    enrollment_matrix = np.stack(data.enrollment_samples, axis=0)
    verifier.fit(enrollment_matrix)

    enroll_distances, _ = verifier.score(enrollment_matrix)
    describe_distances("Enrollment", enroll_distances)

    test_distances: Optional[np.ndarray] = None
    if data.test_samples:
        test_matrix = np.stack(data.test_samples, axis=0)
        test_distances, _ = verifier.score(test_matrix)
        describe_distances("Test", test_distances)
    else:
        print("No test samples in dataset; cannot suggest threshold from test data.")

    if test_distances is not None:
        max_enroll = float(enroll_distances.max())
        min_test = float(test_distances.min())
        suggested = (max_enroll + min_test) / 2.0
        print(f"\nSuggested threshold: {suggested:.4f}")
        if max_enroll > min_test:
            print("WARNING: Enrollment and test distance distributions overlap; consider collecting more data.")
    else:
        # fallback suggestion using enrollment only (e.g., 1 std above max)
        fallback = float(enroll_distances.max() + enroll_distances.std())
        print(f"\nSuggested threshold (fallback): {fallback:.4f}")


if __name__ == "__main__":
    main()
