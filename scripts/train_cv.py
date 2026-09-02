"""
3-fold patient-level cross-validation orchestrator: for each fold, trains a
linear probe from scratch (ungated early stopping), fine-tunes stage 4 from
that fold's own best linear-probe checkpoint (compressed cosine decay,
ungated patience=15, max_epochs=50 -- same recipe validated tonight),
evaluates on the fold's held-out test set, then automatically proceeds to
the next fold with no manual intervention.

Crash-safe: a JSON status file is updated after every epoch of every stage,
so if the process dies partway through an overnight run, the exact
fold/stage/epoch reached is always known from disk, not lost. Each fold is
wrapped in try/except so one fold's failure doesn't stop the others.
"""
import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_split import make_cv_folds  # noqa: E402
from dataset import PneumoDataset  # noqa: E402
from finetune import run_finetune  # noqa: E402
from model import VistaClassifier  # noqa: E402
from train import run_linear_probe, set_all_seeds  # noqa: E402
from threshold_tuning import sweep_thresholds  # noqa: E402
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StatusLogger:
    """Crash-safe status file: overwritten after every epoch with the exact
    (fold, stage, epoch) reached, plus an append-only text log for history."""

    def __init__(self, cv_dir):
        self.status_path = os.path.join(cv_dir, "cv_status.json")
        self.log_path = os.path.join(cv_dir, "cv_master_log.txt")
        self.state = {"folds": {}, "started_at": datetime.now().isoformat()}
        self._write()

    def _write(self):
        with open(self.status_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def log(self, msg):
        line = f"[{datetime.now().isoformat()}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def update(self, fold, stage, epoch=None, val_metrics=None, status="running", extra=None):
        key = str(fold)
        entry = self.state["folds"].setdefault(key, {})
        entry["stage"] = stage
        entry["status"] = status
        entry["last_update"] = datetime.now().isoformat()
        if epoch is not None:
            entry["epoch"] = epoch
        if val_metrics is not None:
            entry["last_val_metrics"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                          for k, v in val_metrics.items()}
        if extra:
            entry.update(extra)
        self._write()


def make_linear_probe_args(seed, patience, max_epochs):
    return argparse.Namespace(
        patch_size=224, batch_size=1, accum_steps=4, amp=True,
        epochs=max_epochs, patience=patience, lr=1e-3, hidden_dim=128, dropout=0.3,
        seed=seed, num_workers=0, selection_metric="auprc",
    )


def make_finetune_args(seed):
    return argparse.Namespace(
        unfreeze_stages=1, patch_size=224, batch_size=1, accum_steps=4, amp=True,
        max_epochs=50, patience=15, encoder_lr=1e-5, head_lr=1e-3,
        warmup_epochs=2, warmup_start_lr=0.0, lr_decay_epochs=22, lr_min_frac=0.1,
        seed=seed, num_workers=0, selection_metric="auprc", time_probe_epochs=0,
    )


def get_probs_for_files(checkpoint_path, filepaths, patch_size=224):
    """Compute model probabilities for an arbitrary explicit file list
    (a fold's own val or test set), not the fixed patient_level_split()."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    ds = PneumoDataset(filepaths, patch_size=(patch_size,) * 3)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = VistaClassifier(freeze_encoder=True, hidden_dim=config.get("hidden_dim", 128),
                             dropout=config.get("dropout", 0.3)).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    all_y, all_p = [], []
    with torch.no_grad():
        for x, y, _paths in loader:
            x = x.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(x)
            probs = torch.sigmoid(logits.float()).cpu()
            all_y.extend(y.tolist())
            all_p.extend(probs.tolist())
    return np.array(all_y), np.array(all_p)


def evaluate_fold(checkpoint_path, splits, patch_size=224):
    """Tune threshold on the fold's own val set, apply (fixed) to the fold's
    held-out test set -- test is touched exactly once, here."""
    y_val, p_val = get_probs_for_files(checkpoint_path, splits["val"], patch_size)
    rows = sweep_thresholds(y_val, p_val)
    best = max(rows, key=lambda r: r["balanced_accuracy"])
    threshold = best["threshold"]

    y_test, p_test = get_probs_for_files(checkpoint_path, splits["test"], patch_size)
    auroc = roc_auc_score(y_test, p_test)
    auprc = average_precision_score(y_test, p_test)
    y_pred = (p_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = accuracy_score(y_test, y_pred)
    bal_acc = (sens + spec) / 2

    return {
        "threshold": threshold, "auroc": auroc, "auprc": auprc,
        "sensitivity": sens, "specificity": spec, "accuracy": acc, "balanced_accuracy": bal_acc,
        "n_test": len(y_test), "n_test_pos": int(y_test.sum()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def run_one_fold(fold, cv_dir, logger, lp_patience, lp_max_epochs, seed):
    fold_idx = fold["fold"]
    splits = fold["splits"]
    fold_dir = os.path.join(cv_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    logger.log(f"=== FOLD {fold_idx}: starting linear probe ===")
    logger.update(fold_idx, "linear_probe", status="running")

    lp_run_dir = os.path.join(fold_dir, "linear_probe")
    lp_args = make_linear_probe_args(seed, lp_patience, lp_max_epochs)

    def lp_progress(epoch, val_metrics):
        logger.update(fold_idx, "linear_probe", epoch=epoch, val_metrics=val_metrics, status="running")

    set_all_seeds(seed)
    lp_result = run_linear_probe(splits, lp_run_dir, lp_args, progress_cb=lp_progress)
    logger.log(f"=== FOLD {fold_idx}: linear probe done, best_epoch={lp_result['best_epoch']}, "
               f"best_score={lp_result['best_score']:.4f} ===")
    logger.update(fold_idx, "linear_probe", status="complete",
                  extra={"linear_probe_best": lp_result})

    logger.log(f"=== FOLD {fold_idx}: starting fine-tune from this fold's own linear probe ===")
    logger.update(fold_idx, "fine_tune", status="running")

    ft_run_dir = os.path.join(fold_dir, "finetune")
    ft_args = make_finetune_args(seed)
    lp_checkpoint = os.path.join(lp_result["ckpt_dir"], "best.pt")

    def ft_progress(epoch, val_metrics):
        logger.update(fold_idx, "fine_tune", epoch=epoch, val_metrics=val_metrics, status="running")

    set_all_seeds(seed)
    ft_result = run_finetune(splits, lp_checkpoint, ft_run_dir, ft_args, progress_cb=ft_progress)
    logger.log(f"=== FOLD {fold_idx}: fine-tune done, best_epoch={ft_result['best_epoch']}, "
               f"best_score={ft_result['best_score']:.4f} ===")
    logger.update(fold_idx, "fine_tune", status="complete", extra={"finetune_best": ft_result})

    logger.log(f"=== FOLD {fold_idx}: evaluating on held-out test set ===")
    logger.update(fold_idx, "evaluate", status="running")

    ft_checkpoint = os.path.join(ft_result["ckpt_dir"], "best.pt")
    test_metrics = evaluate_fold(ft_checkpoint, splits)

    logger.log(f"=== FOLD {fold_idx}: test results -- AUROC={test_metrics['auroc']:.4f} "
               f"AUPRC={test_metrics['auprc']:.4f} sens={test_metrics['sensitivity']:.4f} "
               f"spec={test_metrics['specificity']:.4f} acc={test_metrics['accuracy']:.4f} ===")
    logger.update(fold_idx, "evaluate", status="complete", extra={"test_metrics": test_metrics})

    with open(os.path.join(fold_dir, "fold_summary.json"), "w") as f:
        json.dump({
            "fold": fold_idx, "linear_probe_best": lp_result, "finetune_best": ft_result,
            "test_metrics": test_metrics,
        }, f, indent=2, default=str)

    return test_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lp-patience", type=int, default=15)
    p.add_argument("--lp-max-epochs", type=int, default=60)
    p.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "runs"))
    args = p.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_dir = os.path.join(args.output_dir, f"cv_{timestamp}")
    os.makedirs(cv_dir, exist_ok=True)

    logger = StatusLogger(cv_dir)
    logger.log(f"Starting 3-fold CV run. Output dir: {cv_dir}")

    folds = make_cv_folds(n_splits=args.n_splits, seed=args.seed)
    with open(os.path.join(cv_dir, "folds.json"), "w") as f:
        json.dump([{"fold": fo["fold"], "splits": fo["splits"]} for fo in folds], f, indent=2)
    logger.log(f"Built {len(folds)} folds. Verified patient-level split with no leakage (see cv_split.py checks).")

    all_test_metrics = []
    for fold in folds:
        fold_idx = fold["fold"]
        t0 = time.time()
        try:
            test_metrics = run_one_fold(fold, cv_dir, logger, args.lp_patience, args.lp_max_epochs, args.seed)
            all_test_metrics.append(test_metrics)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.log(f"!!! FOLD {fold_idx} FAILED: {type(e).__name__}: {e}\n{tb}")
            logger.update(fold_idx, "failed", status="failed", extra={"error": str(e), "traceback": tb})
            continue
        elapsed = time.time() - t0
        logger.log(f"=== FOLD {fold_idx} complete in {elapsed/60:.1f} min ===")

    logger.log(f"\n=== ALL FOLDS COMPLETE: {len(all_test_metrics)}/{len(folds)} succeeded ===")

    if all_test_metrics:
        summary = {}
        for metric in ["auroc", "auprc", "sensitivity", "specificity", "accuracy", "balanced_accuracy"]:
            values = [m[metric] for m in all_test_metrics]
            summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                "values": values}
        logger.log("\n=== CROSS-VALIDATED RESULTS (mean +/- std across folds) ===")
        for metric, stats in summary.items():
            logger.log(f"  {metric:20s}: {stats['mean']:.4f} +/- {stats['std']:.4f}  "
                       f"(per-fold: {[f'{v:.4f}' for v in stats['values']]})")

        with open(os.path.join(cv_dir, "cv_final_summary.json"), "w") as f:
            json.dump({"per_fold": all_test_metrics, "summary": summary}, f, indent=2, default=str)
        logger.log(f"\nSaved final summary to {os.path.join(cv_dir, 'cv_final_summary.json')}")
    else:
        logger.log("No folds completed successfully -- no summary to report.")


if __name__ == "__main__":
    main()
