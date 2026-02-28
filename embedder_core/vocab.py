import json
from typing import Any, Dict, List, Tuple

from .constants import SPECIAL_TOKENS, UNK


def normalize_keysym(ev: Dict[str, Any]) -> str:
    """
    Turn an event into a stable keysym string.

    Priorities:
      1) ev["keysym"] if present
      2) ev["char"] (single char)
      3) ev["keycode"] as "kc_XX"
    """
    keysym = ev.get("keysym")
    ch = ev.get("char")
    keycode = ev.get("keycode")

    if keysym is not None and str(keysym).strip() != "":
        s = str(keysym).strip().lower()
        if s in ("return",):
            s = "enter"
        if s in ("space",):
            s = "space"
        return s

    if ch is not None and str(ch) != "":
        s = str(ch)
        if len(s) == 1:
            return s.lower()
        return s.lower()

    if keycode is not None:
        try:
            return f"kc_{int(keycode)}"
        except Exception:
            pass

    return UNK


def build_keysym_vocab(user_json_paths: List[str], min_freq: int = 2) -> Tuple[Dict[str, int], List[str]]:
    """
    Scan all events across all users and build a keysym vocabulary.
    min_freq filters rare keys (helps generalization).
    """
    from collections import Counter

    counter = Counter()
    for path in user_json_paths:
        with open(path, "r", encoding="utf-8") as handle:
            blob = json.load(handle)
        runs = blob.get("enrollment_runs", [])
        for run in runs:
            events = run.get("events", [])
            for ev in events:
                if ev.get("type") in ("keydown", "keyup"):
                    counter[normalize_keysym(ev)] += 1

    vocab = list(SPECIAL_TOKENS)
    for token, freq in counter.most_common():
        if token in SPECIAL_TOKENS:
            continue
        if freq >= min_freq:
            vocab.append(token)

    stoi = {tok: i for i, tok in enumerate(vocab)}
    return stoi, vocab
