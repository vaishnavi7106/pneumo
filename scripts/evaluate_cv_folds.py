"""
Run once AFTER a full 5-fold CV run (train_cv.py) completes. For each fold:

  1. Determine the composite-best epoch checkpoint -- if the fold's finetune
     run already used composite selection, checkpoints/best.pt already IS
     the right one; if it used the older AUPRC-only selection, recompute
     from train_log.csv (every epoch's checkpoint is saved, not just best.pt,
     so this needs no retraining) and use that epoch's checkpoint instead.
  2. Evaluate that checkpoint on the fold's held-out test set -- same
     evaluate_fold logic as train_cv.py (tune threshold on val, apply once
     to test).
  3. Aggregate across folds (mean +/- std), same format as train_cv.py's
     own final summary, saved to <cv_dir>/cv_final_summary_composite.json.

This makes all 5 folds comparable under ONE selection metric even if the CV
run started before the composite metric existed and only later folds used it.

Usage: python scripts/evaluate_cv_folds.py runs/cv_20260907_120514
"""
import argparse
import json
import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reselect_checkpoint import recompute_composite  # noqa: E402
from train_cv import evaluate_fold  # noqa: E402


def pick_best_checkpoint(finetune_dir):
    """Returns (checkpoint_path, epoch, was_reselected)."""
    log_path = os.path.join(finetune_dir, "train_log.csv")
    ckpt_dir = os.path.join(finetune_dir, "checkpoints")
    best_ckpt_path = os.path.join(ckpt_dir, "best.pt")

    df = pd.read_csv(log_path)
    val_df = df[df["split"] == "val"].copy()
    val_df["composite"] = val_df.apply(recompute_composite, axis=1)
    best_row = val_df.loc[val_df["composite"].idxmax()]
    best_epoch = int(best_row["epoch"])

    ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("epoch") == best_epoch:
        # best.pt already IS the composite-best epoch (this fold either used
        # composite selection already, or happens to agree with AUPRC here)
        return best_ckpt_path, best_epoch, False

    src = os.path.join(ckpt_dir, f"epoch_{best_epoch:03d}.pt")
    if not os.path.exists(src):
        print(f"  WARNING: epoch_{best_epoch:03d}.pt missing in {ckpt_dir}, falling back to best.pt "
              f"(epoch {ckpt.get('epoch')})")
        return best_ckpt_path, ckpt.get("epoch"), False

    dst = os.path.join(ckpt_dir, "best_composite.pt")
    shutil.copy2(src, dst)
    return dst, best_epoch, True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cv_dir", help="the cv_<timestamp> directory produced by train_cv.py")
    args = p.parse_args()

    folds_path = os.path.join(args.cv_dir, "folds.json")
    with open(folds_path) as f:
        folds = json.load(f)

    all_test_metrics = []
    checkpoints_used = []
    for fold in folds:
        fold_idx = fold["fold"]
        splits = fold["splits"]
        finetune_dir = os.path.join(args.cv_dir, f"fold_{fold_idx}", "finetune")

        if not os.path.exists(os.path.join(finetune_dir, "train_log.csv")):
            print(f"FOLD {fold_idx}: SKIP -- no completed finetune run found")
            continue

        ckpt_path, best_epoch, was_reselected = pick_best_checkpoint(finetune_dir)
        print(f"FOLD {fold_idx}: using epoch {best_epoch} "
              f"({'re-selected via composite, differs from AUPRC-only best.pt' if was_reselected else 'best.pt already composite-consistent'})")

        test_metrics = evaluate_fold(ckpt_path, splits)
        test_metrics["fold"] = fold_idx
        test_metrics["selected_epoch"] = best_epoch
        all_test_metrics.append(test_metrics)
        checkpoints_used.append({"fold": fold_idx, "checkpoint": ckpt_path, "epoch": best_epoch,
                                  "was_reselected": was_reselected})

        print(f"  TEST: AUROC={test_metrics['auroc']:.4f} AUPRC={test_metrics['auprc']:.4f} "
              f"sens={test_metrics['sensitivity']:.4f} spec={test_metrics['specificity']:.4f} "
              f"acc={test_metrics['accuracy']:.4f}")

    if not all_test_metrics:
        print("\nNo completed folds found -- nothing to summarize.")
        return

    summary = {}
    for metric in ["auroc", "auprc", "sensitivity", "specificity", "accuracy", "balanced_accuracy"]:
        values = [m[metric] for m in all_test_metrics]
        summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}

    print(f"\n=== CROSS-VALIDATED RESULTS ({len(all_test_metrics)} folds, composite-selected "
          f"checkpoints, mean +/- std) ===")
    for metric, stats in summary.items():
        print(f"  {metric:20s}: {stats['mean']:.4f} +/- {stats['std']:.4f}  "
              f"(per-fold: {[f'{v:.4f}' for v in stats['values']]})")

    out_path = os.path.join(args.cv_dir, "cv_final_summary_composite.json")
    with open(out_path, "w") as f:
        json.dump({"per_fold": all_test_metrics, "summary": summary, "checkpoints_used": checkpoints_used},
                   f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
