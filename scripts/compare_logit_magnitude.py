"""
Compare validation-set logit magnitude distributions between the linear
probe (epoch 39) and the fine-tune run's current best checkpoint, to
determine whether elevated fine-tuning val_loss is benign (model becoming
more confident/less-calibrated -- larger |logit| without hurting ranking)
or a real instability signal. Runs on CPU deliberately to avoid contending
with the live GPU fine-tuning job.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PneumoDataset  # noqa: E402
from model import VistaClassifier  # noqa: E402
from split import patient_level_split  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(os.cpu_count() or 8)  # no longer competing with anything -- training run has stopped


def get_logits_cpu(checkpoint_path, unfreeze_stages=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    splits, _ = patient_level_split(seed=config["split_seed"])

    ds = PneumoDataset(splits["val"], patch_size=(config["patch_size"],) * 3)

    model = VistaClassifier(freeze_encoder=True, hidden_dim=config.get("hidden_dim", 128),
                             dropout=config.get("dropout", 0.3))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    logits, labels = [], []
    t0 = time.time()
    for i in range(len(ds)):
        x, y, _ = ds[i]
        x = x.unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        logits.append(out.item())
        labels.append(y.item())
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(ds)} ({time.time()-t0:.0f}s elapsed)")

    return np.array(logits), np.array(labels), ckpt.get("epoch"), ckpt.get("selection_score")


def summarize(name, logits, labels):
    abs_logits = np.abs(logits)
    print(f"\n=== {name} ===")
    print(f"  n={len(logits)}")
    print(f"  mean |logit| = {abs_logits.mean():.3f}, median = {np.median(abs_logits):.3f}, "
          f"std = {abs_logits.std():.3f}")
    print(f"  max |logit| = {abs_logits.max():.3f}, min |logit| = {abs_logits.min():.3f}")
    print(f"  logit range: [{logits.min():.3f}, {logits.max():.3f}]")
    n_extreme = (abs_logits > 10).sum()
    print(f"  |logit| > 10 (very confident): {n_extreme}/{len(logits)}")
    has_nan = np.isnan(logits).any()
    has_inf = np.isinf(logits).any()
    print(f"  NaN: {has_nan}, Inf: {has_inf}")
    return abs_logits.mean(), has_nan, has_inf


def main():
    linear_probe_ckpt = os.path.join(ROOT, "runs", "run_20260901_154953", "checkpoints", "best.pt")

    import glob
    finetune_dirs = sorted(glob.glob(os.path.join(ROOT, "runs", "finetune_*")))
    finetune_ckpt = os.path.join(finetune_dirs[-1], "checkpoints", "best.pt")

    print(f"Linear probe checkpoint: {linear_probe_ckpt}")
    print(f"Fine-tune checkpoint: {finetune_ckpt}")

    cache_path = os.path.join(ROOT, "scripts", "logit_comparison_cache.npz")

    def cached_or_compute(key, ckpt_path):
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            if f"{key}_logits" in data:
                print(f"  (using cached {key} logits from {cache_path})")
                return data[f"{key}_logits"], data[f"{key}_labels"], data[f"{key}_epoch"].item(), data[f"{key}_score"].item()
        logits, labels, epoch, score = get_logits_cpu(ckpt_path)
        # merge into cache file (preserve any other key already saved)
        existing = dict(np.load(cache_path, allow_pickle=True)) if os.path.exists(cache_path) else {}
        existing[f"{key}_logits"] = logits
        existing[f"{key}_labels"] = labels
        existing[f"{key}_epoch"] = epoch
        existing[f"{key}_score"] = score
        np.savez(cache_path, **existing)
        return logits, labels, epoch, score

    print("\nRunning linear-probe (epoch 39) CPU inference on val set...")
    lp_logits, lp_labels, lp_epoch, lp_score = cached_or_compute("lp", linear_probe_ckpt)
    print(f"Linear probe checkpoint epoch={lp_epoch}, score={lp_score}")

    print("\nRunning fine-tune (current best) CPU inference on val set...")
    ft_logits, ft_labels, ft_epoch, ft_score = cached_or_compute("ft", finetune_ckpt)
    print(f"Fine-tune checkpoint epoch={ft_epoch}, score={ft_score}")

    assert np.array_equal(lp_labels, ft_labels), "val label order mismatch between runs -- split changed?"

    lp_mean, lp_nan, lp_inf = summarize(f"LINEAR PROBE (epoch {lp_epoch}, auprc={lp_score:.4f})",
                                         lp_logits, lp_labels)
    ft_mean, ft_nan, ft_inf = summarize(f"FINE-TUNE (epoch {ft_epoch}, auprc={ft_score:.4f})",
                                         ft_logits, ft_labels)

    print(f"\n=== COMPARISON ===")
    print(f"mean |logit|: linear probe={lp_mean:.3f} -> fine-tune={ft_mean:.3f} "
          f"({'+' if ft_mean>=lp_mean else ''}{ft_mean-lp_mean:.3f}, "
          f"{100*(ft_mean/lp_mean - 1):.1f}% change)")
    print(f"Any NaN/Inf anywhere: {lp_nan or lp_inf or ft_nan or ft_inf}")


if __name__ == "__main__":
    main()
