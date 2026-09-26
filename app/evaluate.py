from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import Classifier  # noqa: E402
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    args = parser.parse_args()

    labels = {}
    with open(args.labels_csv) as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = row["true_label"]

    clf = Classifier()
    classes = clf.classes

    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))  # true -> predicted -> count
    low_conf_correct = 0
    low_conf_total = 0

    eval_dir = Path(args.eval_dir)
    for fname, true_label in sorted(labels.items()):
        fp = eval_dir / fname
        if not fp.exists():
            continue
        img = Image.open(fp)
        result = clf.predict(img)
        pred = result["label"]
        confusion[true_label][pred] += 1
        total += 1
        if pred == true_label:
            correct += 1
        if result["low_confidence"]:
            low_conf_total += 1
            if pred == true_label:
                low_conf_correct += 1

    print(f"Overall accuracy on eval_set: {correct}/{total} = {correct/total:.3f}\n")

    print(f"Low-confidence tiles: {low_conf_total}/{total} "
          f"({low_conf_total/total:.1%}) flagged")
    if low_conf_total:
        print(f"  accuracy WITHIN low-confidence tiles: "
              f"{low_conf_correct}/{low_conf_total} = {low_conf_correct/low_conf_total:.3f}")
    print(f"  accuracy on the REST (high-confidence tiles): "
          f"{correct - low_conf_correct}/{total - low_conf_total} = "
          f"{(correct - low_conf_correct)/(total - low_conf_total):.3f}\n")

    print("Confusion matrix (rows = true label, cols = predicted):")
    header = "true\\pred".ljust(14) + "".join(c[:10].ljust(11) for c in classes)
    print(header)
    for true_label in classes:
        row = confusion[true_label]
        print(true_label.ljust(14) + "".join(str(row.get(c, 0)).ljust(11) for c in classes))


if __name__ == "__main__":
    main()
