"""
1-epoch smoke test of the new fixed-FOV pipeline (adaptive ROI crop +
constant 3mm spacing + pad/crop to 224^3, roi_bboxes_adaptive.csv) through
the REAL training scripts (train.py's run_linear_probe, finetune.py's
run_finetune, train_cv.py's evaluate_fold) -- not just dataset.py in
isolation. Confirms the new CLI wiring (--fixed-fov / --roi-bbox-csv on both
train.py and finetune.py) actually works end-to-end before committing to a
real multi-epoch run. Uses the pre-built cache from build_fixed_fov_cache.py,
so this should be fast (no cache misses).

Runs BOTH configs we cached: baseline_1ch and air_mask_2ch.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_split import make_cv_folds  # noqa: E402
from finetune import run_finetune  # noqa: E402
from train import run_linear_probe, set_all_seeds  # noqa: E402
from train_cv import evaluate_fold  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTIVE_ROI_CSV = os.path.join(ROOT, "scripts", "roi_bboxes_adaptive.csv")

ARMS = [
    {"name": "baseline_1ch", "air_mask_threshold": None},
    {"name": "air_mask_2ch", "air_mask_threshold": -600.0},
]


def run_arm(arm, splits, seed, output_dir):
    air_mask_threshold = arm["air_mask_threshold"]
    run_dir = os.path.join(output_dir, f"smoke_fixedfov_{arm['name']}")
    os.makedirs(run_dir, exist_ok=True)

    common = dict(
        patch_size=224, batch_size=1, accum_steps=4, amp=True,
        seed=seed, num_workers=2, selection_metric="composite", augment=False,
        air_mask_threshold=air_mask_threshold, air_mask_classifier_path=None,
        boundary_distance_channel=False, fixed_fov=True, roi_bbox_csv=ADAPTIVE_ROI_CSV,
    )
    lp_args = argparse.Namespace(epochs=1, patience=1, lr=1e-3, hidden_dim=128, dropout=0.3, **common)
    ft_args = argparse.Namespace(
        unfreeze_stages=1, max_epochs=1, patience=1, encoder_lr=1e-5, head_lr=1e-3,
        warmup_epochs=1, warmup_start_lr=0.0, lr_decay_epochs=1, lr_min_frac=0.1, time_probe_epochs=0,
        **common,
    )

    print(f"\n{'='*70}\n=== ARM {arm['name']}: linear probe (smoke, 1 epoch) ===\n{'='*70}")
    set_all_seeds(seed)
    lp_run_dir = os.path.join(run_dir, "linear_probe")
    lp_result = run_linear_probe(splits, lp_run_dir, lp_args)
    print(f"[{arm['name']}] linear probe smoke OK: best_epoch={lp_result['best_epoch']}")

    print(f"\n{'='*70}\n=== ARM {arm['name']}: fine-tune (smoke, 1 epoch) ===\n{'='*70}")
    ft_run_dir = os.path.join(run_dir, "finetune")
    lp_checkpoint = os.path.join(lp_result["ckpt_dir"], "best.pt")
    set_all_seeds(seed)
    ft_result = run_finetune(splits, lp_checkpoint, ft_run_dir, ft_args)
    print(f"[{arm['name']}] fine-tune smoke OK: best_epoch={ft_result['best_epoch']}")

    print(f"\n{'='*70}\n=== ARM {arm['name']}: evaluate ===\n{'='*70}")
    ft_checkpoint = os.path.join(ft_result["ckpt_dir"], "best.pt")
    test_metrics = evaluate_fold(ft_checkpoint, splits, patch_size=224)
    print(f"[{arm['name']}] TEST (smoke, meaningless after 1 epoch): "
          f"AUROC={test_metrics['auroc']:.4f} AUPRC={test_metrics['auprc']:.4f}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "runs"))
    args = p.parse_args()

    folds = make_cv_folds(n_splits=5, seed=args.seed)
    splits = folds[4]["splits"]
    print(f"Fold 4 splits: { {k: len(v) for k, v in splits.items()} }")

    for arm in ARMS:
        ok = run_arm(arm, splits, args.seed, args.output_dir)
        print(f"\n>>> {arm['name']}: {'PASSED' if ok else 'FAILED'} <<<")

    print("\nAll smoke tests completed without crashing.")


if __name__ == "__main__":
    main()
