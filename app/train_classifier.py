"""
Trains the land-use classifier and writes data/model.pkl.

Primary path: train on the real labeled tiles GalaxEye provided
(candidate_tiles/<ClassName>/*.png, a 7-class EuroSAT subset: Forest,
River, Residential, Industrial, AnnualCrop, SeaLake, Highway). Point
--data-dir at the `candidate_tiles` folder from the assignment zip.

Fallback path: if no --data-dir is given, fall back to a small synthesized
dataset (see synthesize_dataset() below) so the repo is still runnable
standalone without the (non-redistributed) dataset attached. This fallback
is honestly weaker -- it exists so `git clone && run` still works, not
because it's how the shipped model was actually trained.

Usage:
    python3 app/train_classifier.py --data-dir /path/to/candidate_tiles
    python3 app/train_classifier.py                # synthetic fallback
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features  # noqa: E402

REAL_CLASSES = ["Forest", "River", "Residential", "Industrial",
                "AnnualCrop", "SeaLake", "Highway"]

# --- synthetic fallback (documented placeholder, see module docstring) ---
PROFILES = {
    "forest":       [(0.18, .05), (0.32, .05), (0.15, .04), (.04,.02),(.04,.02),(.03,.02), (0.14,.03), (0.02,.03), (0.22,.03), (0.020,.008)],
    "water":        [(0.10, .04), (0.16, .04), (0.32, .06), (.02,.01),(.02,.01),(.03,.02), (0.06,.03), (0.19,.04), (0.19,.04), (0.006,.003)],
    "residential":  [(0.45, .07), (0.43, .06), (0.40, .07), (.08,.03),(.07,.03),(.08,.03), (-0.02,.03),(0.01,.03), (0.43,.05), (0.055,.012)],
    "agricultural": [(0.40, .06), (0.45, .06), (0.25, .05), (.05,.02),(.05,.02),(.04,.02), (0.05,.03), (-0.07,.03),(0.37,.04), (0.018,.007)],
    "barren":       [(0.55, .06), (0.48, .06), (0.36, .06), (.04,.02),(.04,.02),(.04,.02), (-0.07,.03),(-0.06,.03),(0.46,.05), (0.015,.006)],
    "cloud":        [(0.85, .05), (0.85, .05), (0.85, .05), (.03,.01),(.03,.01),(.03,.01), (0.00,.02), (0.00,.02), (0.85,.04), (0.008,.004)],
}
N_PER_CLASS = 400
RNG = np.random.default_rng(42)


def synthesize_dataset():
    X, y = [], []
    for cls, profile in PROFILES.items():
        means = np.array([p[0] for p in profile])
        stds = np.array([p[1] for p in profile])
        samples = RNG.normal(loc=means, scale=stds, size=(N_PER_CLASS, len(profile)))
        X.append(samples)
        y += [cls] * N_PER_CLASS
    return np.vstack(X).astype(np.float32), np.array(y)


def load_real_dataset(data_dir: Path):
    X, y = [], []
    for cls in REAL_CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {cls_dir}")
        files = sorted(cls_dir.glob("*.png"))
        for fp in files:
            img = Image.open(fp)
            X.append(extract_features(img))
            y.append(cls)
    return np.vstack(X).astype(np.float32), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None,
                         help="Path to candidate_tiles/ (real, labeled). "
                              "Omit to fall back to synthetic training data.")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
        print(f"Training on real labeled tiles from {data_dir}")
        X, y = load_real_dataset(data_dir)
    else:
        print("No --data-dir given: falling back to SYNTHETIC training data. "
              "Pass --data-dir /path/to/candidate_tiles to train on the real "
              "GalaxEye-provided tiles instead.")
        X, y = synthesize_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    print("\n--- held-out split from training data ---")
    print(classification_report(y_test, clf.predict(X_test)))

    out = Path(__file__).resolve().parent.parent / "data" / "model.pkl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"model": clf, "classes": list(clf.classes_),
                     "trained_on": "real" if args.data_dir else "synthetic"}, f)
    print(f"\nSaved model to {out}")


if __name__ == "__main__":
    main()
