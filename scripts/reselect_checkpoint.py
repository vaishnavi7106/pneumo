"""
Retroactively re-select the "best" checkpoint from an ALREADY-COMPLETED (or
in-progress) fine-tune run, using the composite selection metric, without
retraining -- both linear_probe/ and finetune/ already save a checkpoint for
EVERY epoch (epoch_NNN.pt), not just best.pt, so the AUPRC-only selection
that ran at training time can be overridden after the fact just by reading
train_log.csv and picking a different epoch's already-saved checkpoint.

Use this to fix folds from a 5-fold CV run that started before the composite
metric existed (see train.py's COMPOSITE_WEIGHT_* constants), instead of
discarding and re-running.
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (  # noqa: E402
    COMPOSITE_WEIGHT_AUPRC, COMPOSITE_WEIGHT_AUROC, COMPOSITE_WEIGHT_SENS, COMPOSITE_WEIGHT_SPEC,
)


def recompute_composite(row):
    auroc = row["auroc"] if pd.notna(row["auroc"]) else row["auprc"]
    sens = row["sensitivity"] if pd.notna(row["sensitivity"]) else 0.0
    spec = row["specificity"] if pd.notna(row["specificity"]) else 0.0
    return (COMPOSITE_WEIGHT_AUROC * auroc + COMPOSITE_WEIGHT_AUPRC * row["auprc"]
            + COMPOSITE_WEIGHT_SENS * sens + COMPOSITE_WEIGHT_SPEC * spec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="a linear_probe/ or finetune/ run directory containing "
                                    "train_log.csv and checkpoints/epoch_NNN.pt files")
    args = p.parse_args()

    log_path = os.path.join(args.run_dir, "train_log.csv")
    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    df = pd.read_csv(log_path)
    val_df = df[df["split"] == "val"].copy()
    val_df["composite"] = val_df.apply(recompute_composite, axis=1)

    print(f"{'epoch':>5} {'auroc':>7} {'auprc':>7} {'sens':>7} {'spec':>7} {'composite':>10} "
          f"{'(was AUPRC-selected)' if False else ''}")
    for _, row in val_df.iterrows():
        print(f"{int(row['epoch']):>5} {row['auroc']:>7.4f} {row['auprc']:>7.4f} "
              f"{row['sensitivity']:>7.4f} {row['specificity']:>7.4f} {row['composite']:>10.4f}")

    best_auprc_row = val_df.loc[val_df["auprc"].idxmax()]
    best_composite_row = val_df.loc[val_df["composite"].idxmax()]

    print(f"\nWould have selected under AUPRC-only: epoch {int(best_auprc_row['epoch'])} "
          f"(auprc={best_auprc_row['auprc']:.4f}, sens={best_auprc_row['sensitivity']:.4f}, "
          f"spec={best_auprc_row['specificity']:.4f})")
    print(f"Selects under composite:              epoch {int(best_composite_row['epoch'])} "
          f"(auprc={best_composite_row['auprc']:.4f}, sens={best_composite_row['sensitivity']:.4f}, "
          f"spec={best_composite_row['specificity']:.4f}, composite={best_composite_row['composite']:.4f})")

    best_epoch = int(best_composite_row["epoch"])
    src_ckpt = os.path.join(ckpt_dir, f"epoch_{best_epoch:03d}.pt")
    dst_ckpt = os.path.join(ckpt_dir, "best_composite.pt")

    if not os.path.exists(src_ckpt):
        available = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
        print(f"\nWARNING: {src_ckpt} not found (per-epoch checkpoints may not have been saved "
              f"for every epoch, or this run predates that feature). Available: {available}")
        return

    import shutil
    shutil.copy2(src_ckpt, dst_ckpt)
    print(f"\nCopied {src_ckpt} -> {dst_ckpt}")
    print("Use this checkpoint (not checkpoints/best.pt) for fine-tuning-from / evaluate_fold "
          "on this fold to get the composite-selected model.")


if __name__ == "__main__":
    main()
