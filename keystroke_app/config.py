from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent

PROMPT_QUEUE_SIZE = 5
SENTENCE_DATASET_PATH = APP_ROOT / "medium_english_sentences.txt"

DEFAULT_DATASET_PATH = APP_ROOT / "keystroke_dataset.json"

DATASET_VERSION = 3

DEFAULT_DISTANCE_THRESHOLD = 4.5
LEGACY_SCORE_THRESHOLD = 0.72
