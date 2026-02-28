import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

RAW_RUN_FORMAT = "native_key_events_v1"


@dataclass
class KeyEventRec:
    kind: str
    timestamp_ms: float
    keycode: int
    keysym: str
    char: str


class RunCapture:
    """
    Captures native keydown/keyup keyboard events for one run.
    We capture all keydown/keyup events; text features are derived from printable keydowns.
    """

    def __init__(self):
        self.events: List[KeyEventRec] = []
        self.active_down_counts: Dict[int, int] = {}

    def reset(self):
        self.events.clear()
        self.active_down_counts.clear()

    @staticmethod
    def _event_timestamp_ms(event) -> float:
        native_ts = getattr(event, "time", None)
        if isinstance(native_ts, (int, float)):
            return float(native_ts)
        return float(time.time() * 1000.0)

    def on_key_press(self, event):
        ch = str(getattr(event, "char", ""))
        keycode = int(event.keycode)
        self.events.append(
            KeyEventRec(
                kind="keydown",
                timestamp_ms=self._event_timestamp_ms(event),
                keycode=keycode,
                keysym=str(getattr(event, "keysym", "")),
                char=ch,
            )
        )
        self.active_down_counts[keycode] = int(self.active_down_counts.get(keycode, 0) + 1)

    def on_key_release(self, event):
        keycode = int(event.keycode)
        if int(self.active_down_counts.get(keycode, 0)) <= 0:
            return
        self.events.append(
            KeyEventRec(
                kind="keyup",
                timestamp_ms=self._event_timestamp_ms(event),
                keycode=keycode,
                keysym=str(getattr(event, "keysym", "")),
                char=str(getattr(event, "char", "")),
            )
        )
        remaining = int(self.active_down_counts.get(keycode, 0)) - 1
        if remaining > 0:
            self.active_down_counts[keycode] = remaining
        else:
            self.active_down_counts.pop(keycode, None)

    @staticmethod
    def _timing_stats(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.zeros(6, dtype=np.float32)
        q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90])
        return np.array(
            [
                float(values.mean()),
                float(values.std()),
                float(q10),
                float(q50),
                float(q90),
                float(np.max(values)),
            ],
            dtype=np.float32,
        )

    @classmethod
    def _collect_run_stream_from_events(cls, events: List[Dict[str, Any]]) -> Optional[Tuple[List[str], np.ndarray, np.ndarray]]:
        down_stack: Dict[int, List[int]] = {}
        chars_raw: List[str] = []
        down_times_s: List[float] = []
        dwell_raw: List[Optional[float]] = []

        for ev in events:
            kind = str(ev["type"])
            keycode = int(ev["keycode"])
            t_s = float(ev["timestamp_ms"]) / 1000.0

            if kind == "keydown":
                ch = str(ev.get("char", ""))
                is_text_key = ch == " " or (ch != "" and ch.isprintable())
                if not is_text_key:
                    continue

                chars_raw.append(ch)
                down_times_s.append(t_s)
                dwell_raw.append(None)
                idx = len(chars_raw) - 1
                down_stack.setdefault(keycode, []).append(idx)
                continue

            if kind == "keyup":
                stack = down_stack.get(keycode)
                if not stack:
                    continue
                idx = stack.pop()
                dwell = t_s - down_times_s[idx]
                if 0.005 <= dwell <= 2.0:
                    dwell_raw[idx] = dwell

        dwell_list: List[float] = []
        flight_list: List[float] = []
        chars: List[str] = []

        if not down_times_s:
            return None

        prev_down: Optional[float] = None
        flights_raw: List[float] = []
        for t_down in down_times_s:
            if prev_down is None:
                flights_raw.append(0.0)
            else:
                flights_raw.append(max(0.0, t_down - prev_down))
            prev_down = t_down

        last_was_space = False
        for i, ch in enumerate(chars_raw):
            if ch.isspace():
                if last_was_space:
                    continue
                ch = " "
                last_was_space = True
            else:
                last_was_space = False

            d = dwell_raw[i] if i < len(dwell_raw) and dwell_raw[i] is not None else 0.0
            f = flights_raw[i] if i < len(flights_raw) else 0.0
            dwell_list.append(d)
            flight_list.append(f)
            chars.append(ch)

        if not dwell_list:
            return None

        dwell = np.array(dwell_list, dtype=np.float32)
        flight = np.array(flight_list, dtype=np.float32)

        dwell = np.clip(dwell, 0.0, 2.0)
        flight = np.clip(flight, 0.0, 2.0)
        return chars, dwell, flight

    def _collect_run_stream(self) -> Optional[Tuple[List[str], np.ndarray, np.ndarray]]:
        raw_run = self.build_raw_run()
        if raw_run is None:
            return None
        return self._collect_run_stream_from_events(raw_run["events"])

    @classmethod
    def _feature_vector_from_parts(
        cls,
        chars: List[str],
        dwell: np.ndarray,
        flight: np.ndarray,
    ) -> np.ndarray:
        dwell_stats = cls._timing_stats(dwell)
        flight_stats = cls._timing_stats(flight)
        total_chars = float(len(chars))
        uppercase_ratio = float(sum(c.isalpha() and c.isupper() for c in chars) / max(1.0, total_chars))
        punctuation_ratio = float(sum(c in ".,;:!?" for c in chars) / max(1.0, total_chars))
        space_ratio = float(sum(c == " " for c in chars) / max(1.0, total_chars))
        long_pause_ratio = float((flight > 0.35).mean()) if flight.size else 0.0
        total_dwell = float(dwell.sum())
        total_flight = float(flight.sum())

        feat = np.concatenate(
            [
                np.log1p(dwell_stats),
                np.log1p(flight_stats),
                np.array(
                    [
                        np.log1p(total_chars),
                        uppercase_ratio,
                        punctuation_ratio,
                        space_ratio,
                        long_pause_ratio,
                        np.log1p(total_dwell),
                        np.log1p(total_flight),
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(np.float32)
        return feat

    def build_raw_run(self) -> Optional[Dict[str, Any]]:
        if not self.events:
            return None

        events = [
            {
                "type": ev.kind,
                "timestamp_ms": float(ev.timestamp_ms),
                "keycode": int(ev.keycode),
                "keysym": str(ev.keysym),
                "char": str(ev.char),
            }
            for ev in self.events
        ]
        return {
            "events": events,
        }

    def build_feature_vector(self) -> Optional[np.ndarray]:
        stream = self._collect_run_stream()
        if stream is None:
            return None
        chars, dwell, flight = stream
        return self._feature_vector_from_parts(chars, dwell, flight)


def normalize_raw_run(raw_run: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_run, dict):
        raise ValueError("Raw run must be an object.")

    events = raw_run.get("events")
    if not isinstance(events, list):
        raise ValueError("Raw run must include an 'events' list.")
    if not events:
        raise ValueError("Raw run must include at least one keyboard event.")

    out_events: List[Dict[str, Any]] = []
    last_ts: Optional[float] = None

    for i, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"events[{i}] must be an object.")

        kind = str(event.get("type", ""))
        if kind not in {"keydown", "keyup"}:
            raise ValueError(f"events[{i}].type must be 'keydown' or 'keyup'.")

        timestamp_ms_raw = event.get("timestamp_ms")
        if not isinstance(timestamp_ms_raw, (int, float)):
            raise ValueError(f"events[{i}].timestamp_ms must be numeric.")
        timestamp_ms = float(timestamp_ms_raw)
        if timestamp_ms < 0.0:
            raise ValueError(f"events[{i}].timestamp_ms must be >= 0.")
        if last_ts is not None and timestamp_ms < last_ts:
            raise ValueError("Event timestamps must be non-decreasing within a run.")
        last_ts = timestamp_ms

        keycode_raw = event.get("keycode")
        if not isinstance(keycode_raw, (int, float)):
            raise ValueError(f"events[{i}].keycode must be numeric.")
        keycode = int(keycode_raw)

        keysym = str(event.get("keysym", ""))
        char = str(event.get("char", ""))

        out_events.append(
            {
                "type": kind,
                "timestamp_ms": timestamp_ms,
                "keycode": keycode,
                "keysym": keysym,
                "char": char,
            }
        )

    if not any(ev["type"] == "keydown" for ev in out_events):
        raise ValueError("Raw run must include at least one keydown event.")

    return {
        "events": out_events,
    }


def build_feature_vector_from_raw_run(raw_run: Dict[str, Any]) -> np.ndarray:
    normalized = normalize_raw_run(raw_run)
    stream = RunCapture._collect_run_stream_from_events(normalized["events"])
    if stream is None:
        raise ValueError("Raw run has no usable keydown timing data.")
    chars, dwell, flight = stream
    return RunCapture._feature_vector_from_parts(chars, dwell, flight)
