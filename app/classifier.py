from __future__ import annotations
import pickle
from pathlib import Path
from PIL import Image

try:
    from .features import extract_features
except ImportError:  # allows running scripts directly (evaluate.py, etc.)
    from features import extract_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.pkl"

# Below this confidence, we don't trust the top prediction (see design_note.md
# "handling low-confidence predictions"). Tunable, not a magic constant hidden
# in code -- surfaced here and in the API response.
LOW_CONFIDENCE_THRESHOLD = 0.55


class Classifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.classes = bundle["classes"]

    def predict(self, img: Image.Image) -> dict:
        feats = extract_features(img).reshape(1, -1)
        proba = self.model.predict_proba(feats)[0]
        best_idx = proba.argmax()
        label = self.classes[best_idx]
        confidence = float(proba[best_idx])

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
            "class_probabilities": {
                cls: round(float(p), 4) for cls, p in zip(self.classes, proba)
            },
        }


# Singleton loaded once at process start (cheap: RandomForest, no GPU, no
# big checkpoint) -- see design_note.md for why this model was chosen.
_classifier: Classifier | None = None


def get_classifier() -> Classifier:
    global _classifier
    if _classifier is None:
        _classifier = Classifier()
    return _classifier
