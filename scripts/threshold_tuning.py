"""
Threshold tuning on a trained checkpoint: recompute val-set prediction
probabilities, sweep decision thresholds, report sens/spec/accuracy/balanced
accuracy at each, and find the threshold maximizing balanced accuracy
(equivalent to maximizing Youden's J = sensitivity + specificity - 1).
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import make_dataloaders  # noqa: E402
from model import VistaClassifier  # noqa: E402
from split import patient_level_split  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_val_probs(checkpoint_path, split="val"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    splits, _ = patient_level_split(seed=config["split_seed"])
    loaders = make_dataloaders(
        splits, batch_size=1, patch_size=(config["patch_size"],) * 3, num_workers=0,
    )

    model = VistaClassifier(freeze_encoder=True, hidden_dim=config.get("hidden_dim", 128),
                             dropout=config.get("dropout", 0.3)).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    all_y, all_p, all_paths = [], [], []
    with torch.no_grad():
        for x, y, paths in loaders[split]:
            x = x.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(x)
            probs = torch.sigmoid(logits.float()).cpu()
            all_y.extend(y.tolist())
            all_p.extend(probs.tolist())
            all_paths.extend(paths)

    return np.array(all_y), np.array(all_p), all_paths


def sweep_thresholds(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = accuracy_score(y_true, y_pred)
        bal_acc = (sens + spec) / 2
        youden_j = sens + spec - 1
        rows.append({
            "threshold": t, "sensitivity": sens, "specificity": spec,
            "accuracy": acc, "balanced_accuracy": bal_acc, "youden_j": youden_j,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                    default=os.path.join(ROOT, "runs", "run_20260901_154953", "checkpoints", "best.pt"))
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = p.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    y_true, y_prob, paths = get_val_probs(args.checkpoint, split=args.split)
    print(f"{args.split} set: {len(y_true)} volumes, {int(y_true.sum())} positive, {int((1-y_true).sum())} negative")

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    print(f"AUROC={auroc:.4f}, AUPRC={auprc:.4f} (threshold-independent, unchanged from training log)")

    rows = sweep_thresholds(y_true, y_prob)

    print(f"\n{'threshold':>10s} {'sens':>7s} {'spec':>7s} {'acc':>7s} {'bal_acc':>8s} {'youden_J':>9s}")
    for r in rows:
        marker = ""
        print(f"{r['threshold']:10.2f} {r['sensitivity']:7.3f} {r['specificity']:7.3f} "
              f"{r['accuracy']:7.3f} {r['balanced_accuracy']:8.3f} {r['youden_j']:9.3f}{marker}")

    best_bal = max(rows, key=lambda r: r["balanced_accuracy"])
    best_youden = max(rows, key=lambda r: r["youden_j"])  # same argmax as balanced_accuracy, kept for clarity

    naive = next(r for r in rows if abs(r["threshold"] - 0.5) < 1e-6)
    if naive is None:
        naive_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, naive_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        naive = {"threshold": 0.5, "sensitivity": tp/(tp+fn), "specificity": tn/(tn+fp),
                 "accuracy": accuracy_score(y_true, naive_pred),
                 "balanced_accuracy": (tp/(tp+fn) + tn/(tn+fp))/2}

    print(f"\n=== naive threshold=0.5 ===")
    print(f"  sens={naive['sensitivity']:.3f} spec={naive['specificity']:.3f} "
          f"acc={naive['accuracy']:.3f} bal_acc={naive['balanced_accuracy']:.3f}")

    print(f"\n=== best threshold by balanced accuracy (== Youden's J) ===")
    print(f"  threshold={best_bal['threshold']:.2f}: sens={best_bal['sensitivity']:.3f} "
          f"spec={best_bal['specificity']:.3f} acc={best_bal['accuracy']:.3f} "
          f"bal_acc={best_bal['balanced_accuracy']:.3f} youden_J={best_bal['youden_j']:.3f}")

    print(f"\nImprovement over naive 0.5: "
          f"bal_acc {naive['balanced_accuracy']:.3f} -> {best_bal['balanced_accuracy']:.3f} "
          f"({'+' if best_bal['balanced_accuracy']>=naive['balanced_accuracy'] else ''}"
          f"{best_bal['balanced_accuracy']-naive['balanced_accuracy']:.3f}), "
          f"acc {naive['accuracy']:.3f} -> {best_bal['accuracy']:.3f}")


if __name__ == "__main__":
    main()
