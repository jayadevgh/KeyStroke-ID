from copy import deepcopy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .capture import RAW_RUN_FORMAT, build_feature_vector_from_raw_run, normalize_raw_run
from .config import DEFAULT_DISTANCE_THRESHOLD, LEGACY_SCORE_THRESHOLD


@dataclass
class SessionData:
    enrollment_samples: List[np.ndarray]
    test_samples: List[np.ndarray]
    enrollment_raw_runs: List[Dict[str, Any]]
    test_raw_runs: List[Dict[str, Any]]
    enrollment_mean: Optional[np.ndarray]
    enrollment_inv_cov: Optional[np.ndarray]
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD


def _copy_samples(samples: List[np.ndarray]) -> List[np.ndarray]:
    return [np.array(s, dtype=np.float32, copy=True) for s in samples]


def _copy_raw_runs(raw_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [deepcopy(run) for run in raw_runs]


def _serialize_raw_runs(raw_runs: List[Dict[str, Any]], field_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, run in enumerate(raw_runs):
        try:
            out.append(normalize_raw_run(run))
        except Exception as ex:
            raise ValueError(f"{field_name}[{i}] is invalid: {ex}") from ex
    return out


def _deserialize_raw_runs(raw: Any, field_name: str) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    if raw is None:
        raise ValueError(f"'{field_name}' is required.")
    if not isinstance(raw, list):
        raise ValueError(f"'{field_name}' must be a list.")

    out_runs: List[Dict[str, Any]] = []
    out_features: List[np.ndarray] = []
    for i, item in enumerate(raw):
        try:
            run = normalize_raw_run(item)
            feat = build_feature_vector_from_raw_run(run)
        except Exception as ex:
            raise ValueError(f"{field_name}[{i}] is not a valid native key event run: {ex}") from ex
        out_runs.append(run)
        out_features.append(np.asarray(feat, dtype=np.float32))
    return out_runs, out_features


def _deserialize_vector(raw: Any, field_name: str) -> Optional[np.ndarray]:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"'{field_name}' must be a non-empty 1D numeric vector.")
    return arr


def _deserialize_matrix(raw: Any, field_name: str) -> Optional[np.ndarray]:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"'{field_name}' must be a non-empty 2D numeric matrix.")
    return arr


def _feature_dim(data: SessionData) -> Optional[int]:
    if data.enrollment_mean is not None:
        return int(data.enrollment_mean.shape[0])
    if data.enrollment_samples:
        return int(data.enrollment_samples[0].shape[0])
    if data.test_samples:
        return int(data.test_samples[0].shape[0])
    return None


def _validate_dimensions(data: SessionData):
    dim = _feature_dim(data)
    if dim is None:
        return
    if data.enrollment_mean is not None and int(data.enrollment_mean.shape[0]) != dim:
        raise ValueError("Enrollment mean dimension does not match dataset dimension.")
    if data.enrollment_inv_cov is not None and data.enrollment_inv_cov.shape != (dim, dim):
        raise ValueError("Enrollment inverse covariance must be square and match dataset dimension.")
    for i, arr in enumerate(data.enrollment_samples):
        if int(arr.shape[0]) != dim:
            raise ValueError(f"Enrollment sample #{i + 1} has wrong feature dimension.")
    for i, arr in enumerate(data.test_samples):
        if int(arr.shape[0]) != dim:
            raise ValueError(f"Test sample #{i + 1} has wrong feature dimension.")
    for i, run in enumerate(data.enrollment_raw_runs):
        feat = build_feature_vector_from_raw_run(run)
        if int(feat.shape[0]) != dim:
            raise ValueError(f"Enrollment raw run #{i + 1} has wrong derived feature dimension.")
    for i, run in enumerate(data.test_raw_runs):
        feat = build_feature_vector_from_raw_run(run)
        if int(feat.shape[0]) != dim:
            raise ValueError(f"Test raw run #{i + 1} has wrong derived feature dimension.")


def _compute_enrollment_stats(
    enrollment_samples: List[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not enrollment_samples:
        return None, None

    X = np.stack(enrollment_samples, axis=0).astype(np.float32)
    mean = X.mean(axis=0)

    if X.shape[0] < 3:
        return mean, None

    cov = np.cov(X.T)
    cov = np.asarray(cov, dtype=np.float32)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    dim = X.shape[1]
    cov = cov + np.eye(dim, dtype=np.float32) * 1e-4
    inv_cov = np.linalg.inv(cov).astype(np.float32)
    return mean, inv_cov


def _normalize_distance_threshold(raw: Any) -> float:
    if not isinstance(raw, (int, float)):
        return float(DEFAULT_DISTANCE_THRESHOLD)
    value = float(raw)
    if value <= 0.0:
        return float(DEFAULT_DISTANCE_THRESHOLD)
    if abs(value - LEGACY_SCORE_THRESHOLD) < 1e-6:
        return float(DEFAULT_DISTANCE_THRESHOLD)
    return value


def session_data_to_payload(data: SessionData, dataset_version: int) -> Dict[str, Any]:
    _validate_dimensions(data)
    feature_dim = _feature_dim(data)
    enrollment_raw = _serialize_raw_runs(data.enrollment_raw_runs, "enrollment_raw_runs")
    test_raw = _serialize_raw_runs(data.test_raw_runs, "test_raw_runs")
    if len(data.enrollment_samples) != len(enrollment_raw):
        raise ValueError("Enrollment sample count must match enrollment raw run count.")
    if len(data.test_samples) != len(test_raw):
        raise ValueError("Test sample count must match test raw run count.")

    payload: Dict[str, Any] = {
        "type": "keystroke_dataset",
        "version": int(dataset_version),
        "created_unix": int(time.time()),
        "run_format": RAW_RUN_FORMAT,
        "distance_threshold": float(data.distance_threshold),
        "feature_dim": int(feature_dim) if feature_dim is not None else None,
        "num_enrollment_runs": int(len(data.enrollment_samples)),
        "num_test_runs": int(len(data.test_samples)),
        "num_enrollment_raw_runs": int(len(enrollment_raw)),
        "num_test_raw_runs": int(len(test_raw)),
        "enrollment_runs": enrollment_raw,
        "test_runs": test_raw,
        "enrollment_mean": None if data.enrollment_mean is None else data.enrollment_mean.astype(float).tolist(),
        "enrollment_inv_cov": None
        if data.enrollment_inv_cov is None
        else data.enrollment_inv_cov.astype(float).tolist(),
    }
    return payload


def payload_to_session_data(payload: Dict[str, Any]) -> SessionData:
    run_format = payload.get("run_format")
    if run_format != RAW_RUN_FORMAT:
        raise ValueError(
            f"Unsupported run_format '{run_format}'. Expected '{RAW_RUN_FORMAT}'."
        )

    enrollment_runs_raw = payload.get("enrollment_runs")
    test_runs_raw = payload.get("test_runs")
    enrollment_raw_runs, enrollment_samples = _deserialize_raw_runs(enrollment_runs_raw, "enrollment_runs")
    test_raw_runs, test_samples = _deserialize_raw_runs(test_runs_raw, "test_runs")

    enrollment_mean = _deserialize_vector(payload.get("enrollment_mean"), "enrollment_mean")
    if enrollment_mean is None and payload.get("reference") is not None:
        enrollment_mean = _deserialize_vector(payload.get("reference"), "reference")

    enrollment_inv_cov = _deserialize_matrix(payload.get("enrollment_inv_cov"), "enrollment_inv_cov")

    distance_threshold_raw = payload.get("distance_threshold", payload.get("score_threshold"))
    distance_threshold = _normalize_distance_threshold(distance_threshold_raw)

    data = SessionData(
        enrollment_samples=enrollment_samples,
        test_samples=test_samples,
        enrollment_raw_runs=enrollment_raw_runs,
        test_raw_runs=test_raw_runs,
        enrollment_mean=enrollment_mean,
        enrollment_inv_cov=enrollment_inv_cov,
        distance_threshold=float(distance_threshold),
    )
    _validate_dimensions(data)

    feature_dim = payload.get("feature_dim")
    if isinstance(feature_dim, int):
        dim = _feature_dim(data)
        if dim is not None and dim != feature_dim:
            raise ValueError(f"Dataset feature_dim ({feature_dim}) does not match data vectors ({dim}).")

    if data.enrollment_mean is None or data.enrollment_inv_cov is None:
        mean, inv_cov = _compute_enrollment_stats(data.enrollment_samples)
        if data.enrollment_mean is None:
            data.enrollment_mean = mean
        if data.enrollment_inv_cov is None:
            data.enrollment_inv_cov = inv_cov

    return data


def save_session_data(path: Path, data: SessionData, dataset_version: int):
    payload = session_data_to_payload(data, dataset_version)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_session_data(path: Path) -> SessionData:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload_to_session_data(payload)


def merge_session_data(base: SessionData, incoming: SessionData) -> SessionData:
    merged_enrollment = _copy_samples(base.enrollment_samples) + _copy_samples(incoming.enrollment_samples)
    merged_test = _copy_samples(base.test_samples) + _copy_samples(incoming.test_samples)
    merged_enrollment_raw = _copy_raw_runs(base.enrollment_raw_runs) + _copy_raw_runs(incoming.enrollment_raw_runs)
    merged_test_raw = _copy_raw_runs(base.test_raw_runs) + _copy_raw_runs(incoming.test_raw_runs)

    if merged_enrollment:
        mean, inv_cov = _compute_enrollment_stats(merged_enrollment)
    else:
        mean = None
        inv_cov = None
        if incoming.enrollment_mean is not None:
            mean = np.array(incoming.enrollment_mean, dtype=np.float32, copy=True)
        elif base.enrollment_mean is not None:
            mean = np.array(base.enrollment_mean, dtype=np.float32, copy=True)
        if incoming.enrollment_inv_cov is not None:
            inv_cov = np.array(incoming.enrollment_inv_cov, dtype=np.float32, copy=True)
        elif base.enrollment_inv_cov is not None:
            inv_cov = np.array(base.enrollment_inv_cov, dtype=np.float32, copy=True)

    merged_threshold = float(DEFAULT_DISTANCE_THRESHOLD)
    merged = SessionData(
        enrollment_samples=merged_enrollment,
        test_samples=merged_test,
        enrollment_raw_runs=merged_enrollment_raw,
        test_raw_runs=merged_test_raw,
        enrollment_mean=mean,
        enrollment_inv_cov=inv_cov,
        distance_threshold=merged_threshold,
    )
    _validate_dimensions(merged)
    return merged


def merge_session_data_file(path: Path, incoming: SessionData, dataset_version: int) -> SessionData:
    if path.exists():
        base = load_session_data(path)
    else:
        base = SessionData(
            enrollment_samples=[],
            test_samples=[],
            enrollment_raw_runs=[],
            test_raw_runs=[],
            enrollment_mean=None,
            enrollment_inv_cov=None,
            distance_threshold=4.5,
        )
    merged = merge_session_data(base, incoming)
    save_session_data(path, merged, dataset_version)
    return merged
