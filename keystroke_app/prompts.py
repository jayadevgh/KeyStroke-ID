import random
from pathlib import Path
from typing import List, Optional, Sequence


def load_sentence_bank(path: Path) -> List[str]:
    sentences = [" ".join(line.strip().split()) for line in path.read_text(encoding="utf-8").splitlines()]
    sentences = [s for s in sentences if s]
    if len(sentences) < 1000:
        raise ValueError(f"Expected at least 1000 sentences in {path}, found {len(sentences)}")
    return sentences


class PromptQueue:
    def __init__(self, sentence_bank: Sequence[str], size: int):
        if size < 1:
            raise ValueError("Prompt queue size must be >= 1.")
        self._sentence_bank = list(sentence_bank)
        if not self._sentence_bank:
            raise ValueError("Sentence bank cannot be empty.")
        self._size = size
        self._queue: List[str] = []
        self._last_served: Optional[str] = None
        self.reset()

    def _random_sentence(self, avoid: Optional[str] = None) -> str:
        if len(self._sentence_bank) == 1:
            return self._sentence_bank[0]
        sentence = random.choice(self._sentence_bank)
        if avoid is None:
            return sentence
        attempts = 0
        while sentence == avoid and attempts < 10:
            sentence = random.choice(self._sentence_bank)
            attempts += 1
        return sentence

    def reset(self):
        self._last_served = None
        self._queue = [self._random_sentence() for _ in range(self._size)]

    def current(self) -> str:
        if not self._queue:
            self.reset()
        sentence = self._queue[0]
        self._last_served = sentence
        return sentence

    def advance(self):
        if not self._queue:
            self.reset()
            return
        self._queue.pop(0)
        self._queue.append(self._random_sentence(self._last_served))
        if self._queue and self._last_served is not None and self._queue[0] == self._last_served:
            self._queue[0] = self._random_sentence(self._last_served)

    def as_text(self) -> str:
        return "\n".join(self._queue)
