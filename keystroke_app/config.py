from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent

PROMPT_QUEUE_SIZE = 5
SENTENCE_DATASET_PATH = APP_ROOT / "medium_english_sentences.txt"

DEFAULT_DATASET_PATH = APP_ROOT / "keystroke_dataset.json"
DEFAULT_EMBEDDER_FINAL_WEIGHTS_PATH = APP_ROOT / "keystroke_user_classifier_keysym_final_weights.pt"
DEFAULT_EMBEDDER_CHECKPOINT_PATH = APP_ROOT / "keystroke_user_classifier_keysym.pt"

DATASET_VERSION = 3

DEFAULT_DISTANCE_THRESHOLD = 50.0
LEGACY_SCORE_THRESHOLD = 0.72
