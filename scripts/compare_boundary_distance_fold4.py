"""
Controlled single-fold comparison: a NEW aux-channel approach, separate from
compare_air_mask_fold4.py (deliberately not overwriting it -- these are two
different hypotheses about how to give the model a free-air-relevant channel).

Motivation: the raw air-mask channel (HU threshold + body mask) gave a small
but real, twice-replicated improvement (fold-4 comparison and the isolated
no-augment 5-fold CV both showed +AUROC/+AUPRC). The stage-2 classifier
refinement of that same mask made things WORSE (see fold4_air_mask_comparison
results: refined arm dropped specificity 0.667->0.400) because its ~24%
held-out recall meant reweighting suppressed most real free-air signal along
with the false positives.

This tries a different idea instead of patching the same one further: replace
the binary/refined air MASK entirely with a smooth, unthresholded geometric
prior -- each voxel's physical distance (mm, normalized) to the body-wall
boundary (see dataset.py's _compute_boundary_distance_channel). No HU
thresholding of "is this air" at all; the density channel already carries
that information, and free air's actual distinguishing property is largely
positional (collects near the peritoneal boundary) rather than purely a
density fact bowel gas doesn't also share. This sidesteps the density-based
false-positive problem instead of trying to classify our way out of it.

All arms use the SAME fold-4 split and SAME training recipe as
compare_air_mask_fold4.py, so results are directly comparable across both
scripts:

  arm A: baseline_1ch            -- in_channels=1 (original, no aux channel)
  arm B: air_mask_2ch            -- in_channels=2, raw air_mask_threshold=-600 HU
                                     (repeated here as the reference point already
                                     established in compare_air_mask_fold4.py)
  arm C: boundary_distance_2ch   -- in_channels=2, smooth distance-to-body-wall
                                     field, no thresholding

Run sequentially (one GPU). Only the aux channel differs between arms.
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
    {"name": "baseline_1ch", "air_mask_threshold": None, "boundary_distance_channel": False},
    {"name": "air_mask_2ch", "air_mask_threshold": -600.0, "boundary_distance_channel": False},
    {"name": "boundary_distance_2ch", "air_mask_threshold": None, "boundary_distance_channel": True},
]


def run_arm(arm, splits, seed, output_dir, lp_patience, lp_max_epochs, batch_size, accum_steps, num_workers):
    air_mask_threshold = arm["air_mask_threshold"]
    boundary_distance_channel = arm["boundary_distance_channel"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"fold4_{arm['name']}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    lp_args = argparse.Namespace(
        patch_size=224, batch_size=batch_size, accum_steps=accum_steps, amp=True,
        epochs=lp_max_epochs, patience=lp_patience, lr=1e-3, hidden_dim=128, dropout=0.3,
        seed=seed, num_workers=num_workers, selection_metric="composite", augment=True,
        air_mask_threshold=air_mask_threshold, air_mask_classifier_path=None,
        boundary_distance_channel=boundary_distance_channel,
    )
    ft_args = argparse.Namespace(
        unfreeze_stages=1, patch_size=224, batch_size=batch_size, accum_steps=accum_steps, amp=True,
        max_epochs=50, patience=15, encoder_lr=1e-5, head_lr=1e-3,
        warmup_epochs=2, warmup_start_lr=0.0, lr_decay_epochs=22, lr_min_frac=0.1,
        seed=seed, num_workers=num_workers, selection_metric="composite", time_probe_epochs=0, augment=True,
        air_mask_threshold=air_mask_threshold, air_mask_classifier_path=None,
        boundary_distance_channel=boundary_distance_channel,
    )

    print(f"\n{'='*70}\n=== ARM {arm['name']} "
          f"(air_mask_threshold={air_mask_threshold}, boundary_distance_channel={boundary_distance_channel}): "
          f"linear probe ===\n{'='*70}")
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
    p.add_argument("--batch-size", type=int, default=1,
                    help="physical batch size for both linear-probe and fine-tune stages "
                         "(default 1, tuned for a 12GB card; a 24GB card can go higher, e.g. 2-4)")
    p.add_argument("--accum-steps", type=int, default=4,
                    help="gradient accumulation steps (effective batch = batch_size * accum_steps)")
    p.add_argument("--num-workers", type=int, default=0,
                    help="dataloader worker processes (0 = load in the main process)")
    args = p.parse_args()

    folds = make_cv_folds(n_splits=5, seed=args.seed)
    splits = folds[4]["splits"]
    print(f"Fold 4 splits: { {k: len(v) for k, v in splits.items()} }")
    print(f"batch_size={args.batch_size}, accum_steps={args.accum_steps} "
          f"(effective batch={args.batch_size * args.accum_steps}), num_workers={args.num_workers}")

    results = {}
    for arm in ARMS:
        results[arm["name"]] = run_arm(arm, splits, args.seed, args.output_dir,
                                        args.lp_patience, args.lp_max_epochs,
                                        args.batch_size, args.accum_steps, args.num_workers)

    print(f"\n{'='*70}\n=== COMPARISON: all arms, fold 4, identical recipe ===\n{'='*70}")
    base = results["baseline_1ch"]["test_metrics"]
    arm_names = [a["name"] for a in ARMS]
    for metric in ["auroc", "auprc", "sensitivity", "specificity", "accuracy"]:
        line = f"  {metric:14s}: " + "  ".join(
            f"{name}={results[name]['test_metrics'][metric]:.4f}"
            + (f" (delta={results[name]['test_metrics'][metric] - base[metric]:+.4f})"
               if name != "baseline_1ch" else "")
            for name in arm_names
        )
        print(line)

    summary_path = os.path.join(args.output_dir, "fold4_boundary_distance_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved comparison summary to {summary_path}")


if __name__ == "__main__":
    main()
