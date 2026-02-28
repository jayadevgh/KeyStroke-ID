import numpy as np

from keystroke_app.verifier import Verifier


def main():
    rng = np.random.default_rng(42)
    feature_dim = 19
    true_mean = np.linspace(-0.5, 0.5, feature_dim, dtype=np.float32)

    enrollment = true_mean + rng.normal(scale=0.5, size=(30, feature_dim)).astype(np.float32)

    verifier = Verifier()
    verifier.fit(enrollment)

    genuine = true_mean + rng.normal(scale=0.5, size=(3, feature_dim)).astype(np.float32)
    impostor = (true_mean + 2.0) + rng.normal(scale=0.5, size=(3, feature_dim)).astype(np.float32)

    samples = np.vstack([genuine, impostor]).astype(np.float32)
    distances, _ = verifier.score(samples)

    print("Distances (first 3 genuine, last 3 impostor):")
    for idx, dist in enumerate(distances):
        label = "genuine" if idx < 3 else "impostor"
        print(f"  Sample {idx + 1} ({label}): {dist:.4f}")

    threshold = (distances[:3].max() + distances[3:].min()) / 2.0
    verifier.distance_threshold = float(threshold)
    _, inlier = verifier.score(samples)

    assert np.all(inlier[:3]), "Expected genuine samples to be inliers"
    assert not np.any(inlier[3:]), "Expected impostor samples to be outliers"

    try:
        verifier.fit(true_mean.reshape(1, -1))
    except ValueError:
        pass
    else:
        raise AssertionError("fit should raise ValueError for fewer than 3 runs")


if __name__ == "__main__":
    main()
