"""
Ensemble check: average validation-set probabilities from the fine-tune
checkpoint and the linear-probe checkpoint, compare AUROC/AUPRC and
threshold-tuned metrics against the standalone fine-tune result.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from threshold_tuning import get_val_probs, sweep_thresholds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def report(name, y_true, y_prob):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    rows = sweep_thresholds(y_true, y_prob)
    best = max(rows, key=lambda r: r["balanced_accuracy"])
    print(f"\n=== {name} ===")
    print(f"AUROC={auroc:.4f} AUPRC={auprc:.4f}")
    print(f"best threshold={best['threshold']:.2f}: sens={best['sensitivity']:.3f} "
          f"spec={best['specificity']:.3f} acc={best['accuracy']:.3f} bal_acc={best['balanced_accuracy']:.3f}")
    return auroc, auprc, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--finetune-checkpoint", type=str,
                    default=os.path.join(ROOT, "runs", "finetune_20260902_140547", "checkpoints", "best.pt"))
    p.add_argument("--linear-probe-checkpoint", type=str,
                    default=os.path.join(ROOT, "runs", "run_20260901_154953", "checkpoints", "best.pt"))
    args = p.parse_args()

    print(f"Fine-tune checkpoint: {args.finetune_checkpoint}")
    print(f"Linear-probe checkpoint: {args.linear_probe_checkpoint}")

    print("\nComputing fine-tune val probabilities...")
    y_true_ft, y_prob_ft, paths_ft = get_val_probs(args.finetune_checkpoint)

    print("Computing linear-probe val probabilities...")
    y_true_lp, y_prob_lp, paths_lp = get_val_probs(args.linear_probe_checkpoint)

    assert paths_ft == paths_lp, "val set order mismatch between checkpoints -- splits differ?"
    assert np.array_equal(y_true_ft, y_true_lp)

    y_true = y_true_ft
    y_prob_ensemble = (y_prob_ft + y_prob_lp) / 2.0

    ft_auroc, ft_auprc, ft_best = report("STANDALONE FINE-TUNE (epoch 5)", y_true, y_prob_ft)
    lp_auroc, lp_auprc, lp_best = report("STANDALONE LINEAR PROBE (epoch 39)", y_true, y_prob_lp)
    en_auroc, en_auprc, en_best = report("ENSEMBLE (avg of both)", y_true, y_prob_ensemble)

    print(f"\n=== COMPARISON (validation set) ===")
    print(f"{'model':30s} {'AUROC':>8s} {'AUPRC':>8s} {'bal_acc(tuned)':>15s} {'acc(tuned)':>12s}")
    print(f"{'fine-tune (epoch 5)':30s} {ft_auroc:8.4f} {ft_auprc:8.4f} "
          f"{ft_best['balanced_accuracy']:15.4f} {ft_best['accuracy']:12.4f}")
    print(f"{'linear probe (epoch 39)':30s} {lp_auroc:8.4f} {lp_auprc:8.4f} "
          f"{lp_best['balanced_accuracy']:15.4f} {lp_best['accuracy']:12.4f}")
    print(f"{'ensemble (avg)':30s} {en_auroc:8.4f} {en_auprc:8.4f} "
          f"{en_best['balanced_accuracy']:15.4f} {en_best['accuracy']:12.4f}")

    winner = max(
        [("fine-tune", ft_auprc, ft_best), ("ensemble", en_auprc, en_best)],
        key=lambda x: x[1],
    )
    print(f"\nWinner on validation AUPRC: {winner[0]} (AUPRC={winner[1]:.4f}, "
          f"tuned threshold={winner[2]['threshold']:.2f})")


if __name__ == "__main__":
    main()
