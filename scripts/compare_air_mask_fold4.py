"""
Controlled single-fold comparison: does adding the air-mask channel help,
holding everything else constant?

Both arms use the SAME fold-4 split (cv_split.make_cv_folds(n_splits=5,
seed=42)[4], identical to the 5-fold CV baseline and the earlier
stage3+4-augmented test) and the SAME training recipe (train-only 3D
augmentation, stage-4-only fine-tuning -- the config just pushed as
train_cv.py's default for the 4090 run):

  arm A: baseline_1ch      -- in_channels=1 (original, no air mask)
  arm B: air_mask_2ch      -- in_channels=2, air_mask_threshold=-600 HU

Run sequentially (one GPU). Only the air-mask channel differs between arms,
so any AUROC/AUPRC difference is attributable to that one change.
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

ARMS = [
    {"name": "baseline_1ch", "air_mask_threshold": None},
    {"name": "air_mask_2ch", "air_mask_threshold": -600.0},
]


def run_arm(arm, splits, seed, output_dir, lp_patience, lp_max_epochs):
    air_mask_threshold = arm["air_mask_threshold"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"fold4_{arm['name']}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    lp_args = argparse.Namespace(
        patch_size=224, batch_size=1, accum_steps=4, amp=True,
        epochs=lp_max_epochs, patience=lp_patience, lr=1e-3, hidden_dim=128, dropout=0.3,
        seed=seed, num_workers=0, selection_metric="auprc", augment=True,
        air_mask_threshold=air_mask_threshold,
    )
    ft_args = argparse.Namespace(
        unfreeze_stages=1, patch_size=224, batch_size=1, accum_steps=4, amp=True,
        max_epochs=50, patience=15, encoder_lr=1e-5, head_lr=1e-3,
        warmup_epochs=2, warmup_start_lr=0.0, lr_decay_epochs=22, lr_min_frac=0.1,
        seed=seed, num_workers=0, selection_metric="auprc", time_probe_epochs=0, augment=True,
        air_mask_threshold=air_mask_threshold,
    )

    print(f"\n{'='*70}\n=== ARM {arm['name']} (air_mask_threshold={air_mask_threshold}): linear probe ===\n{'='*70}")
    set_all_seeds(seed)
    lp_run_dir = os.path.join(run_dir, "linear_probe")
    lp_result = run_linear_probe(splits, lp_run_dir, lp_args)
    print(f"[{arm['name']}] linear probe done: best_epoch={lp_result['best_epoch']}, "
          f"best_score={lp_result['best_score']:.4f}")

    print(f"\n{'='*70}\n=== ARM {arm['name']}: fine-tune ===\n{'='*70}")
    ft_run_dir = os.path.join(run_dir, "finetune")
    lp_checkpoint = os.path.join(lp_result["ckpt_dir"], "best.pt")
    set_all_seeds(seed)
    ft_result = run_finetune(splits, lp_checkpoint, ft_run_dir, ft_args)
    print(f"[{arm['name']}] fine-tune done: best_epoch={ft_result['best_epoch']}, "
          f"best_score={ft_result['best_score']:.4f}")

    print(f"\n{'='*70}\n=== ARM {arm['name']}: evaluating on held-out fold-4 test set ===\n{'='*70}")
    ft_checkpoint = os.path.join(ft_result["ckpt_dir"], "best.pt")
    test_metrics = evaluate_fold(ft_checkpoint, splits)
    print(f"[{arm['name']}] TEST: AUROC={test_metrics['auroc']:.4f} AUPRC={test_metrics['auprc']:.4f} "
          f"sens={test_metrics['sensitivity']:.4f} spec={test_metrics['specificity']:.4f} "
          f"acc={test_metrics['accuracy']:.4f}")

    return {"run_dir": run_dir, "linear_probe_best": lp_result, "finetune_best": ft_result,
            "test_metrics": test_metrics}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lp-patience", type=int, default=15)
    p.add_argument("--lp-max-epochs", type=int, default=60)
    p.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "runs"))
    args = p.parse_args()

    folds = make_cv_folds(n_splits=5, seed=args.seed)
    splits = folds[4]["splits"]
    print(f"Fold 4 splits: { {k: len(v) for k, v in splits.items()} }")

    results = {}
    for arm in ARMS:
        results[arm["name"]] = run_arm(arm, splits, args.seed, args.output_dir,
                                        args.lp_patience, args.lp_max_epochs)

    print(f"\n{'='*70}\n=== COMPARISON: baseline (1ch) vs. air-mask (2ch), fold 4, identical recipe ===\n{'='*70}")
    base = results["baseline_1ch"]["test_metrics"]
    mask = results["air_mask_2ch"]["test_metrics"]
    for metric in ["auroc", "auprc", "sensitivity", "specificity", "accuracy"]:
        delta = mask[metric] - base[metric]
        print(f"  {metric:14s}: baseline_1ch={base[metric]:.4f}  air_mask_2ch={mask[metric]:.4f}  "
              f"delta={delta:+.4f}")

    summary_path = os.path.join(args.output_dir, "fold4_air_mask_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved comparison summary to {summary_path}")


if __name__ == "__main__":
    main()
