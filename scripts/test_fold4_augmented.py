"""
Targeted test on fold 4 ONLY (the worst-performing fold in the 5-fold CV run:
baseline AUROC=0.5730, AUPRC=0.4793, specificity=0.5111 -- barely above
chance) before committing to another full 18-hour 5-fold run.

Changes under test vs. the baseline CV run:
  1. Train-only 3D augmentation (flip/rotate/intensity jitter, augment.py),
     verified train-only (verify_augmentation_split.py) and visually QC'd for
     ROI-crop interaction bugs (qc_augmentation.py) beforehand.
  2. Unfreeze stage 3+4 during fine-tuning (--unfreeze-stages 2) instead of
     stage 4 only.

Reuses cv_split.make_cv_folds(n_splits=5, seed=42) so fold 4's train/val/test
patient split is IDENTICAL to the baseline run -- any difference in the
result is attributable only to the two changes above, not a different split.
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_split import make_cv_folds  # noqa: E402
from finetune import run_finetune  # noqa: E402
from train import run_linear_probe, set_all_seeds  # noqa: E402
from train_cv import evaluate_fold  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE = {"auroc": 0.5730, "auprc": 0.4793, "sensitivity": 0.5714,
            "specificity": 0.5111, "accuracy": 0.5342}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lp-patience", type=int, default=15)
    p.add_argument("--lp-max-epochs", type=int, default=60)
    p.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "runs"))
    args = p.parse_args()

    folds = make_cv_folds(n_splits=5, seed=args.seed)
    fold4 = folds[4]
    splits = fold4["splits"]
    print(f"Fold 4 splits: {{k: len(v) for k, v in splits.items()}} = "
          f"{ {k: len(v) for k, v in splits.items()} }")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"fold4_augmented_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    lp_args = argparse.Namespace(
        patch_size=224, batch_size=1, accum_steps=4, amp=True,
        epochs=args.lp_max_epochs, patience=args.lp_patience, lr=1e-3, hidden_dim=128, dropout=0.3,
        seed=args.seed, num_workers=0, selection_metric="auprc", augment=True,
    )
    ft_args = argparse.Namespace(
        unfreeze_stages=2, patch_size=224, batch_size=1, accum_steps=4, amp=True,
        max_epochs=50, patience=15, encoder_lr=1e-5, head_lr=1e-3,
        warmup_epochs=2, warmup_start_lr=0.0, lr_decay_epochs=22, lr_min_frac=0.1,
        seed=args.seed, num_workers=0, selection_metric="auprc", time_probe_epochs=0, augment=True,
    )

    print("\n=== FOLD 4 (AUGMENTED, unfreeze stage 3+4): linear probe ===")
    set_all_seeds(args.seed)
    lp_run_dir = os.path.join(run_dir, "linear_probe")
    lp_result = run_linear_probe(splits, lp_run_dir, lp_args)
    print(f"Linear probe done: best_epoch={lp_result['best_epoch']}, best_score={lp_result['best_score']:.4f}")

    print("\n=== FOLD 4 (AUGMENTED, unfreeze stage 3+4): fine-tune ===")
    ft_run_dir = os.path.join(run_dir, "finetune")
    lp_checkpoint = os.path.join(lp_result["ckpt_dir"], "best.pt")
    set_all_seeds(args.seed)
    ft_result = run_finetune(splits, lp_checkpoint, ft_run_dir, ft_args)
    print(f"Fine-tune done: best_epoch={ft_result['best_epoch']}, best_score={ft_result['best_score']:.4f}")

    print("\n=== FOLD 4 (AUGMENTED, unfreeze stage 3+4): evaluating on held-out test set ===")
    ft_checkpoint = os.path.join(ft_result["ckpt_dir"], "best.pt")
    test_metrics = evaluate_fold(ft_checkpoint, splits)

    print("\n=== COMPARISON: fold 4 baseline vs. augmented+stage3+4 ===")
    for metric in ["auroc", "auprc", "sensitivity", "specificity", "accuracy"]:
        base = BASELINE[metric]
        new = test_metrics[metric]
        delta = new - base
        print(f"  {metric:14s}: baseline={base:.4f}  new={new:.4f}  delta={delta:+.4f}")

    with open(os.path.join(run_dir, "fold4_augmented_summary.json"), "w") as f:
        json.dump({"baseline": BASELINE, "new": test_metrics,
                   "linear_probe_best": lp_result, "finetune_best": ft_result}, f, indent=2, default=str)
    print(f"\nSaved summary to {os.path.join(run_dir, 'fold4_augmented_summary.json')}")


if __name__ == "__main__":
    main()
