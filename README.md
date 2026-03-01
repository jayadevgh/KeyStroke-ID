# KeyGuard

Continuous keystroke-based identity verification system using embedding-based behavioral modeling.

---

## Table of Contents

- [System Overview](#system-overview)
- [High-Level Architecture](#high-level-architecture)
- [Layer 1: Keystroke Capture](#1-keystroke-capture-layer)
- [Layer 2: Feature Extraction](#2-feature-extraction-layer)
- [Layer 3: Sliding Window Aggregation](#3-sliding-window-aggregation)
- [Layer 4: Embedding Model](#4-embedding-model)
- [Layer 5: Enrollment](#5-enrollment)
- [Layer 6: Live Inference](#6-live-inference)
- [Layer 7: Policy Engine](#7-policy-engine)
- [Layer 8: Multi-User Support](#8-multi-user-support)
- [Layer 9: Step-Up Authentication](#9-step-up-authentication-trigger)
- [Code Structure](#code-structure)
- [Developer Setup](#developer-setup)
- [Configuration](#configuration-parameters)
- [Performance Considerations](#performance-considerations)
- [Security & Storage](#security--storage)
- [Extensibility](#extensibility)

---

## System Overview

KeyGuard is a real-time keystroke dynamics authentication system built around a shared neural embedding model. It operates entirely on-device, requires no cloud dependency at inference time, and introduces zero friction for legitimate users.

The system captures keystroke timing events at the OS level, transforms them into structured feature windows, projects those windows into a learned embedding space, and compares the resulting embeddings against enrolled identity vectors. If behavioral similarity drops below a configurable threshold for a sustained period, a step-up authentication challenge is triggered.

**What KeyGuard is not:**
- It is not a keylogger. Key content is never stored or transmitted.
- It is not cloud-dependent. All inference runs locally.
- It is not a replacement for existing authentication. It is an additional continuous verification layer.

The system is structured into nine primary layers:

1. Keystroke Capture Layer
2. Feature Extraction Layer
3. Sliding Window Aggregation
4. Embedding Model
5. Enrollment
6. Live Inference
7. Policy Engine
8. Multi-User Support
9. Step-Up Authentication Trigger

---

## High-Level Architecture

```
Raw Keystroke Events (key down / key up timestamps)
                ↓
Timing Feature Extraction (dwell, flight, digraph latency)
                ↓
Sliding Window Builder (5s non-overlapping windows)
                ↓
Feature Normalization & Tensor Construction
                ↓
Shared Embedding Model (neural network)
                ↓
Live Embedding Vector (e.g. 64D or 128D)
                ↓
Similarity Scoring (cosine / euclidean vs. enrolled embeddings)
                ↓
Policy Engine (3 consecutive anomaly threshold)
                ↓
Step-Up Authentication Trigger (Touch ID / PIN)
```

---

## 1. Keystroke Capture Layer

### Purpose

Collect raw keyboard event timestamps at the OS level without intercepting or storing key content.

### Implementation

The capture layer registers a global keyboard hook using OS-level event listeners. On macOS, this uses the Quartz event tap API or `pynput`. On Windows, a low-level keyboard hook via `win32api` or `pynput` is used.

```python
# Example using pynput
from pynput import keyboard

def on_press(key):
    record_event(key_id=hash(key), event_type='down', timestamp=time.perf_counter())

def on_release(key):
    record_event(key_id=hash(key), event_type='up', timestamp=time.perf_counter())

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
```

### Captured Data (in-memory only)

| Field        | Type    | Description                        |
|--------------|---------|------------------------------------|
| `key_id`     | int     | Hashed key identifier (not content)|
| `event_type` | str     | `'down'` or `'up'`                 |
| `timestamp`  | float   | `time.perf_counter()` in seconds   |

Raw events are held in a short-lived in-memory ring buffer. They are never written to disk. The key identifier is hashed so that actual key content is not recoverable from the stored data.

### Derived Signals

From consecutive raw timestamps, three primary timing signals are computed:

```
dwell_time        = key_up[i]   - key_down[i]         # how long a key is held
flight_time       = key_down[i] - key_up[i-1]         # gap between consecutive keys
digraph_latency   = key_down[i] - key_down[i-1]       # onset-to-onset timing
```

These three signals form the basis of all downstream feature computation.

---

## 2. Feature Extraction Layer

### Purpose

Transform raw timing events into structured per-keystroke feature vectors suitable for model input.

### Input

Sequence of timestamped key events from the capture buffer.

### Output

A structured feature vector per keystroke event.

```python
# Per-event feature vector
{
    "dwell_time":          float,   # ms
    "flight_time":         float,   # ms
    "digraph_latency":     float,   # ms
    "rolling_avg_dwell":   float,   # rolling mean over last N events
    "rolling_std_dwell":   float,   # rolling std over last N events
    "rolling_avg_flight":  float,   # rolling mean over last N events
    "rolling_std_flight":  float,   # rolling std over last N events
}
```

### Rolling Statistics

Rolling statistics (mean and std) are computed over a configurable lookback window (default: last 10 events). These capture local rhythm — whether the user is typing fast or slow in a given stretch — which is a strong behavioral signal independent of absolute speed.

### Normalization

Before windowing, features are normalized using per-user statistics computed during enrollment (z-score normalization):

```python
normalized = (x - enrollment_mean) / enrollment_std
```

This makes the model robust to absolute speed differences between sessions and users.

### Implementation Note

Feature extraction runs in streaming mode. There is no batch delay. Each key event produces an updated feature vector immediately.

---

## 3. Sliding Window Aggregation

### Purpose

Aggregate streaming per-event features into fixed-size windows for model input.

### Window Configuration

| Parameter          | Default | Description                          |
|--------------------|---------|--------------------------------------|
| `WINDOW_SIZE_SEC`  | 5       | Duration of each window in seconds   |
| `WINDOW_STRIDE_SEC`| 5       | Stride between windows (non-overlapping by default) |
| `MIN_EVENTS`       | 10      | Minimum key events required to score a window |

Windows with fewer than `MIN_EVENTS` keystrokes are discarded and do not count toward the anomaly threshold. This prevents false positives during pauses or idle periods.

### Window Output Formats

Two output formats are supported depending on embedding architecture:

**Aggregated statistical vector** (for feedforward networks):

```python
window_vector = [
    mean_dwell,
    std_dwell,
    mean_flight,
    std_flight,
    mean_digraph,
    std_digraph,
    key_event_density,    # events per second
    p25_dwell,            # 25th percentile dwell
    p75_dwell,            # 75th percentile dwell
    p25_flight,           # 25th percentile flight
    p75_flight,           # 75th percentile flight
]
```

**Ordered sequence tensor** (for temporal models like LSTM or Transformer):

```python
window_tensor.shape = (sequence_length, feature_dim)
# e.g. (50, 7) for 50 events with 7 features each
```

The architecture flag in config determines which format is produced.

---

## 4. Embedding Model

### Purpose

Map variable typing windows into a fixed-dimensional embedding space where identity is geometrically separable.

### Model Architecture Options

| Architecture        | Input Format          | Notes                                      |
|---------------------|-----------------------|--------------------------------------------|
| Feedforward (MLP)   | Aggregated vector     | Fastest inference, simplest to deploy      |
| 1D CNN              | Sequence tensor       | Captures local temporal patterns           |
| LSTM / GRU          | Sequence tensor       | Captures longer-range temporal dependencies|
| Transformer encoder | Sequence tensor       | Highest capacity, slower inference         |

**Current implementation:** lightweight feedforward network (MLP).

### Model Definition (example)

```python
import torch
import torch.nn as nn

class KeystrokeEmbedder(nn.Module):
    def __init__(self, input_dim=11, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim),
        )
        self.norm = nn.functional.normalize

    def forward(self, x):
        embedding = self.encoder(x)
        return self.norm(embedding, dim=-1)  # L2-normalized
```

Output embeddings are L2-normalized so that cosine similarity is equivalent to dot product, simplifying scoring.

### Training Objective

Two approaches are supported:

**Classification loss** (simpler, works well with sufficient users):

```python
# Adds a classification head during training, discarded at inference
loss = CrossEntropyLoss(embeddings_projected, user_labels)
```

**Triplet loss** (better metric learning, preferred for small datasets):

```python
# Anchor, positive (same user), negative (different user)
loss = TripletMarginLoss(anchor, positive, negative, margin=0.3)
```

Triplet loss explicitly optimizes for same-user embeddings clustering together and different-user embeddings being pushed apart, which directly aligns with inference-time behavior.

### Training Data Requirements

- Minimum: 5-10 users, ~10 minutes of typing each
- Recommended: 50+ users, diverse typing contexts (coding, prose, forms)
- Augmentation: apply small timing perturbations to simulate natural variation

### Model Output

Fixed-dimension embedding vector, e.g. shape `(64,)`, L2-normalized.

---

## 5. Enrollment

### Purpose

Build a stable per-user identity embedding from a short natural typing session.

### Process

```python
def enroll_user(user_id: str, duration_seconds: int = 120):
    embeddings = []

    # Collect windows during enrollment session
    for window in collect_windows(duration=duration_seconds):
        if window.event_count >= MIN_EVENTS:
            embedding = model.encode(window.to_tensor())
            embeddings.append(embedding)

    # Average embeddings into a single stable identity vector
    identity_embedding = torch.stack(embeddings).mean(dim=0)
    identity_embedding = F.normalize(identity_embedding, dim=-1)

    # Store encrypted
    store.save(user_id, identity_embedding)
```

### Stored Enrollment Record

```python
{
    "user_id":              str,      # unique identifier
    "identity_embedding":   list,     # float32 vector, e.g. 64D
    "enrollment_timestamp": str,      # ISO 8601
    "window_count":         int,      # number of windows averaged
    "mean_dwell":           float,    # used for normalization at inference
    "std_dwell":            float,
    "mean_flight":          float,
    "std_flight":           float,
}
```

Normalization statistics (mean/std) computed during enrollment are stored alongside the embedding and applied to all future inference windows for this user.

### Storage

Embeddings are stored in a locally encrypted file (AES-256). The encryption key is derived from the device's hardware identity or a local keychain entry. No enrollment data is transmitted.

---

## 6. Live Inference

### Purpose

Continuously score the current typing session against enrolled identity embeddings.

### Inference Loop

```python
def inference_loop():
    while session_active:
        window = window_builder.get_next_window()

        if window is None or window.event_count < MIN_EVENTS:
            continue  # skip idle or sparse windows

        # Normalize using enrolled user's statistics
        normalized = normalize(window.to_tensor(), enrolled_stats)

        # Embed
        live_embedding = model.encode(normalized)

        # Score against all enrolled profiles
        scores = {}
        for user_id, stored_embedding in enrollment_store.items():
            scores[user_id] = cosine_similarity(live_embedding, stored_embedding)

        best_match_user = max(scores, key=scores.get)
        best_match_score = scores[best_match_user]

        policy_engine.update(best_match_score, best_match_user)
```

### Similarity Metrics

| Metric               | Formula                                    | Notes                              |
|----------------------|--------------------------------------------|------------------------------------|
| Cosine similarity    | `dot(a, b) / (norm(a) * norm(b))`          | Default; range [-1, 1]             |
| Euclidean distance   | `norm(a - b)`                              | Lower = more similar               |
| Learned metric       | Trained distance head                      | Future extension                   |

With L2-normalized embeddings, cosine similarity reduces to a dot product and is the preferred metric.

---

## 7. Policy Engine

### Purpose

Apply the anomaly detection threshold and manage the consecutive strike counter.

### Logic

```python
class PolicyEngine:
    def __init__(self, threshold=0.72, required_strikes=3):
        self.threshold = threshold
        self.required_strikes = required_strikes
        self.strike_count = 0
        self.current_identity = None

    def update(self, similarity_score: float, matched_user: str):
        if similarity_score >= self.threshold:
            self.strike_count = 0                    # reset on match
            self.current_identity = matched_user
            self.emit_signal("VERIFIED", matched_user, similarity_score)
        else:
            self.strike_count += 1
            self.emit_signal("UNCERTAIN", matched_user, similarity_score)

            if self.strike_count >= self.required_strikes:
                self.strike_count = 0
                self.emit_signal("ANOMALY_DETECTED")
                step_up_auth.trigger()

    def emit_signal(self, status: str, user: str = None, score: float = None):
        # Update UI confidence banner and log event
        ...
```

### Identity Confidence States

| State     | Condition                             | UI Signal |
|-----------|---------------------------------------|-----------|
| VERIFIED  | similarity ≥ threshold                | Green     |
| UNCERTAIN | similarity < threshold, strikes < 3   | Yellow    |
| ANOMALY   | 3 consecutive uncertain windows       | Red       |

### Threshold Tuning

The default threshold of `0.72` is empirically derived. To tune for a specific deployment:

1. Collect a held-out validation set of legitimate and impostor sessions
2. Plot ROC curve of true positive rate vs. false positive rate across threshold values
3. Select threshold at desired operating point (e.g. FAR < 1%, FRR < 5%)

---

## 8. Multi-User Support

### Purpose

Support multiple enrolled users on the same device and identify which enrolled user is currently typing.

### Enrollment Store

```python
enrollment_store = {
    "user_001": {"embedding": [...], "stats": {...}},
    "user_002": {"embedding": [...], "stats": {...}},
    ...
}
```

### Inference Behavior

At each window, the live embedding is compared against all enrolled profiles simultaneously:

```python
scores = {
    user_id: cosine_similarity(live_embedding, stored_embedding)
    for user_id, stored_embedding in enrollment_store.items()
}

best_match = max(scores, key=scores.get)
best_score = scores[best_match]

if best_score >= ANOMALY_THRESHOLD:
    identity = best_match         # known user identified
else:
    identity = "UNKNOWN"          # no enrolled user matches
```

### Use Cases Enabled

- **Shared workstations:** multiple enrolled users, system identifies who is at the keyboard at any time
- **Session handoff:** Profile 2 sits down after Profile 1, system transitions identity automatically after Touch ID verification
- **Impostor detection:** live embedding matches no enrolled user above threshold

---

## 9. Step-Up Authentication Trigger

### Purpose

Prompt re-authentication when the policy engine detects a sustained anomaly.

### Trigger Flow

```python
def trigger_step_up():
    result = os_auth.prompt()    # Touch ID or PIN via OS API

    if result == AUTH_SUCCESS:
        policy_engine.reset_strikes()
        session.continue_active()
        # Optional: re-enroll from current window to refresh baseline
    else:
        session.lock()
        audit_log.record("FAILED_STEP_UP", timestamp=now())
```

### OS Integration

| Platform | Method                                      |
|----------|---------------------------------------------|
| macOS    | `LocalAuthentication` framework via PyObjC  |
| Windows  | Windows Hello API via `ctypes` / `win32security` |
| Linux    | PAM challenge via `python-pam`              |

The step-up module is intentionally decoupled from the rest of the pipeline. Any re-authentication mechanism can be plugged in by implementing the `AuthProvider` interface.

---

## Code Structure

```
keyguard/
│
├── capture/
│   ├── keyboard_listener.py     # OS-level keyboard hook, pynput wrapper
│   ├── event_buffer.py          # In-memory ring buffer for raw events
│   └── event_types.py           # KeyEvent dataclass definition
│
├── features/
│   ├── timing_features.py       # Dwell, flight, digraph computation
│   ├── window_builder.py        # Sliding window aggregation
│   ├── normalizer.py            # Z-score normalization using enrollment stats
│   └── tensor_utils.py          # Feature vector / tensor construction
│
├── model/
│   ├── embedding_model.py       # KeystrokeEmbedder nn.Module definition
│   ├── train.py                 # Training loop, triplet/classification loss
│   ├── inference.py             # Model loading, encode() wrapper
│   └── checkpoints/             # Saved model weights (.pt files)
│
├── enrollment/
│   ├── enroll_user.py           # Enrollment session logic
│   ├── enrollment_store.py      # Encrypted read/write for identity embeddings
│   └── schema.py                # EnrollmentRecord dataclass
│
├── scoring/
│   ├── similarity.py            # Cosine, euclidean metric implementations
│   └── threshold_policy.py      # PolicyEngine: strike counter, state machine
│
├── security/
│   ├── step_up_auth.py          # AuthProvider interface + OS implementations
│   └── session_manager.py       # Session lock / continue logic
│
├── storage/
│   ├── embedding_store.py       # AES-256 encrypted local storage
│   └── audit_log.py             # Tamper-evident event log
│
├── ui/
│   ├── confidence_banner.py     # Green / yellow / red identity signal UI
│   └── enrollment_ui.py         # Enrollment session prompt
│
├── config.py                    # All configurable parameters
├── main.py                      # Entry point, wires all layers together
└── tests/
    ├── test_features.py
    ├── test_policy_engine.py
    ├── test_enrollment.py
    └── test_inference.py
```

Each module is decoupled behind a clean interface to allow model swapping, threshold tuning, and deployment configuration without touching other layers.

---

## Developer Setup

### Requirements

- Python 3.10+
- macOS 12+ or Windows 10+ (for Touch ID / Windows Hello integration)
- No GPU required at inference time

### Installation

```bash
git clone https://github.com/your-org/keyguard.git
cd keyguard

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Dependencies

```
pynput>=1.7.6          # keyboard capture
torch>=2.0.0           # embedding model
numpy>=1.24.0          # feature computation
cryptography>=41.0.0   # AES-256 embedding storage
pyobjc>=9.0            # macOS Touch ID (macOS only)
pam>=0.2.0             # Linux PAM auth (Linux only)
```

### Running the System

**Enroll a user:**

```bash
python -m keyguard.enrollment.enroll_user --user-id kumar --duration 120
```

**Start live monitoring:**

```bash
python main.py --user-id kumar
```

**Train the embedding model** (requires a labeled keystroke dataset):

```bash
python -m keyguard.model.train \
    --data-path ./data/typing_dataset.csv \
    --epochs 50 \
    --embedding-dim 64 \
    --loss triplet
```

### Running Tests

```bash
pytest tests/ -v
```

---

## Configuration Parameters

All parameters are defined in `config.py` and can be overridden via environment variables or a `keyguard.yaml` config file.

```python
# Windowing
WINDOW_SIZE_SECONDS   = 5       # Duration of each scoring window
WINDOW_STRIDE_SECONDS = 5       # Stride between windows
MIN_EVENTS_PER_WINDOW = 10      # Minimum keystrokes to score a window

# Policy
ANOMALY_THRESHOLD     = 0.72    # Cosine similarity threshold (0.0 - 1.0)
CONSECUTIVE_STRIKES   = 3       # Consecutive anomalous windows before trigger

# Model
EMBEDDING_DIM         = 64      # Output embedding dimension
MODEL_PATH            = "model/checkpoints/embedder.pt"

# Storage
EMBEDDING_STORE_PATH  = "~/.keyguard/embeddings.enc"
AUDIT_LOG_PATH        = "~/.keyguard/audit.log"

# Auth
AUTH_PROVIDER         = "touchid"   # "touchid" | "pin" | "pam"
```

---

## Performance Considerations

| Metric                  | Target      | Notes                                        |
|-------------------------|-------------|----------------------------------------------|
| Inference latency       | < 50ms      | Per 5s window on CPU                         |
| Memory footprint        | < 50MB      | Model + buffers at runtime                   |
| CPU usage (idle)        | < 2%        | Background monitoring mode                   |
| Enrollment time         | ~2 minutes  | Minimum for reliable embedding               |
| Model size              | < 5MB       | MLP with 64D embedding                       |

Inference runs on CPU. No GPU is required or assumed. The MLP architecture is chosen specifically for its low inference latency. If a temporal architecture (LSTM, Transformer) is substituted, latency should be re-benchmarked.

---

## Security & Storage

- **No keystroke content stored** — only timing-derived features, never character content
- **No cloud dependency** — all inference and storage is local
- **AES-256 encryption** — identity embeddings encrypted at rest
- **Hashed key identifiers** — key content is not recoverable from stored data
- **Tamper-evident audit log** — all step-up events, session locks, and anomaly detections are logged with timestamps
- **Optional federated training** — future mode for improving the shared model across organizations without sharing raw data

---

## Extensibility

The architecture is designed for clean extension at each layer:

| Extension                        | Where to implement              |
|----------------------------------|---------------------------------|
| Swap embedding model             | `model/embedding_model.py`      |
| Add mouse dynamics features      | `features/timing_features.py`   |
| Add online embedding refinement  | `enrollment/enrollment_store.py`|
| Add centralized analytics        | `storage/audit_log.py`          |
| Add new auth provider            | `security/step_up_auth.py`      |
| Adjust anomaly policy            | `scoring/threshold_policy.py`   |

The `AuthProvider` interface in `step_up_auth.py` accepts any callable that returns `AUTH_SUCCESS` or `AUTH_FAILED`, making it straightforward to swap in a different re-authentication mechanism.

---

## Built With

Python · PyTorch · pynput · cryptography · PyObjC (macOS Touch ID)

## Team

Built at **Hack to the Future – Presented by AISC @ UW** by Jayadev, Jiahe, Sambhu, and Owen.