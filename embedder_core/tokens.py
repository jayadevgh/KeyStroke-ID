from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .constants import UNK
from .vocab import normalize_keysym


@dataclass
class KeypressToken:
    keysym_id: int
    dwell_ms: float
    flight_ms: float


def parse_run_to_tokens(events: List[Dict[str, Any]], stoi: Dict[str, int]) -> List[KeypressToken]:
    """
    Pair keydown->keyup by keycode. Emit keypress tokens ordered by keydown time.

    Token identity uses normalize_keysym on the keyup event (or falls back).
    """
    down_time: Dict[int, float] = {}
    down_keysym: Dict[int, str] = {}

    keypresses: List[Tuple[float, int, float, int]] = []
    # (down_ts, keycode, up_ts, keysym_id)

    for ev in events:
        ev_type = ev.get("type")
        ts = float(ev.get("timestamp_ms"))
        keycode = ev.get("keycode")
        if keycode is None:
            continue
        keycode = int(keycode)

        if ev_type == "keydown":
            down_time[keycode] = ts
            down_keysym[keycode] = normalize_keysym(ev)
        elif ev_type == "keyup":
            if keycode not in down_time:
                continue
            down_ts = down_time.pop(keycode)
            keysym = normalize_keysym(ev)
            if keysym == UNK and keycode in down_keysym:
                keysym = down_keysym[keycode]

            keysym_id = stoi.get(keysym, stoi[UNK])
            keypresses.append((down_ts, keycode, ts, keysym_id))
            down_keysym.pop(keycode, None)

    keypresses.sort(key=lambda row: row[0])

    tokens: List[KeypressToken] = []
    prev_down: Optional[float] = None
    for down_ts, _, up_ts, keysym_id in keypresses:
        dwell = max(0.0, up_ts - down_ts)
        flight = 0.0 if prev_down is None else max(0.0, down_ts - prev_down)
        prev_down = down_ts
        tokens.append(KeypressToken(keysym_id, dwell, flight))

    return tokens
