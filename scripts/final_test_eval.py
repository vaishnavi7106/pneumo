"""
Final held-out test-set evaluation: first and only time the test split is
touched. Uses the winning model (standalone fine-tune, epoch 5) and the
validation-tuned decision threshold (not re-tuned on test).
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from threshold_tuning import get_val_probs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                    default=os.path.join(ROOT, "runs", "finetune_20260902_140547", "checkpoints", "best.pt"))
    p.add_argument("--threshold", type=float, default=0.63,
                    help="decision threshold tuned on validation -- NOT re-tuned here")
    args = p.parse_args()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Using validation-tuned threshold: {args.threshold} (fixed, not re-tuned on test)")
    print("\n*** FIRST AND ONLY TOUCH OF THE HELD-OUT TEST SET ***\n")

    y_true, y_prob, paths = get_val_probs(args.checkpoint, split="test")
    print(f"Test set: {len(y_true)} volumes, {int(y_true.sum())} positive, {int((1-y_true).sum())} negative")

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    y_pred = (y_prob >= args.threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = accuracy_score(y_true, y_pred)
    bal_acc = (sens + spec) / 2

    print(f"\n=== FINAL TEST-SET RESULTS (threshold={args.threshold}) ===")
    print(f"AUROC:            {auroc:.4f}")
    print(f"AUPRC:            {auprc:.4f}")
    print(f"Sensitivity:      {sens:.4f}")
    print(f"Specificity:      {spec:.4f}")
    print(f"Accuracy:         {acc:.4f}")
    print(f"Balanced accuracy:{bal_acc:.4f}")
    print(f"\nConfusion matrix: TP={tp} FP={fp} TN={tn} FN={fn}")

    print(f"\n=== comparison: val (tuning set) vs test (held-out) ===")
    print(f"{'metric':20s} {'val (epoch5)':>15s} {'test':>10s}")
    print(f"{'AUROC':20s} {0.7749:15.4f} {auroc:10.4f}")
    print(f"{'AUPRC':20s} {0.7951:15.4f} {auprc:10.4f}")
    print(f"{'sensitivity':20s} {0.667:15.4f} {sens:10.4f}")
    print(f"{'specificity':20s} {0.879:15.4f} {spec:10.4f}")
    print(f"{'accuracy':20s} {0.796:15.4f} {acc:10.4f}")


if __name__ == "__main__":
    main()
