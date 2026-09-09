"""
Step 9: linear-probe training loop for pneumoperitoneum classification.

Frozen VISTA3D encoder + trainable classification head, AMP fp16, gradient
accumulation. Supports --smoke-test for a short verification run (a few
optimizer updates, checked for NaNs/unexpected encoder grads/memory
stability) that must pass before launching the real, long run.

Run `--smoke-test` first. Only run without it for the actual training job.
"""
import argparse
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime

# cosmetic-only: MONAI's Orientation deprecation notice and torch's
# GradScaler API-migration notice, neither affects correctness of this script
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, roc_auc_score,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import make_dataloaders  # noqa: E402
from model import VistaClassifier  # noqa: E402
from split import patient_level_split  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {
        "python_random_seed": seed, "numpy_seed": seed,
        "torch_manual_seed": seed, "torch_cuda_manual_seed_all": seed,
    }


# Weights for the "composite" selection metric (see compute_metrics below).
# AUPRC-only selection can pick a degenerate checkpoint that ranks a handful
# of confident cases well (inflating AUPRC on a small val set) while barely
# predicting positive at all (e.g. sensitivity=0.18) -- a near-useless model
# for a screening task. Sensitivity is weighted above specificity because
# missing a real pneumoperitoneum (false negative) is the clinically worse
# error for this task. Tune these constants directly if the balance needs
# adjusting; they intentionally aren't a CLI surface since the "right"
# balance is a judgment call, not a per-run experiment variable.
COMPOSITE_WEIGHT_AUROC = 0.25
COMPOSITE_WEIGHT_AUPRC = 0.35
COMPOSITE_WEIGHT_SENS = 0.25
COMPOSITE_WEIGHT_SPEC = 0.15


def compute_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auroc"] = float("nan")  # only one class present in this split
    metrics["auprc"] = average_precision_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # recall / TPR
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # TNR
    metrics["balanced_accuracy"] = (
        (metrics["sensitivity"] + metrics["specificity"]) / 2
        if not (np.isnan(metrics["sensitivity"]) or np.isnan(metrics["specificity"])) else float("nan")
    )

    # composite: penalizes exactly the failure mode where AUPRC/AUROC look
    # good but sens/spec reveal the model has collapsed toward one class.
    # Falls back to auprc-only if auroc/sens/spec are NaN (e.g. a batch/split
    # with only one class present), so it never crashes checkpoint selection.
    auroc_term = metrics["auroc"] if not np.isnan(metrics["auroc"]) else metrics["auprc"]
    sens_term = metrics["sensitivity"] if not np.isnan(metrics["sensitivity"]) else 0.0
    spec_term = metrics["specificity"] if not np.isnan(metrics["specificity"]) else 0.0
    metrics["composite"] = (
        COMPOSITE_WEIGHT_AUROC * auroc_term + COMPOSITE_WEIGHT_AUPRC * metrics["auprc"]
        + COMPOSITE_WEIGHT_SENS * sens_term + COMPOSITE_WEIGHT_SPEC * spec_term
    )
    return metrics


def build_optimizer(model: VistaClassifier, lr: float):
    """Confirm the optimizer sees ONLY trainable (head) parameters -- the
    frozen encoder's params must never appear here, both because they have
    requires_grad=False (so grads would be None) and to keep optimizer state
    (Adam moments) from ever being allocated for 175M dead parameters."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_names_that_leaked = [n for n in trainable_names if n.startswith("encoder.")]
    assert not frozen_names_that_leaked, (
        f"BUG: encoder params found in optimizer's trainable set: {frozen_names_that_leaked[:5]}"
    )
    assert all(n.startswith("head.") for n in trainable_names), (
        f"unexpected trainable params outside head: "
        f"{[n for n in trainable_names if not n.startswith('head.')]}"
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Optimizer trainable params: {len(trainable_params)} tensors, {n_trainable:,} scalars "
          f"(all under 'head.' prefix, confirmed)")
    return torch.optim.Adam(trainable_params, lr=lr)


def run_epoch(model, loader, optimizer, scaler, criterion, device, accum_steps: int,
              train: bool, amp: bool, epoch: int = 0, epochs: int = 0,
              step_counter=None, on_optimizer_step=None):
    """step_counter: optional mutable [int] shared across epochs, incremented once
    per optimizer.step() (i.e. per accumulation boundary), so callers can track a
    global step count across the whole run (not reset per epoch).
    on_optimizer_step: optional callback(global_step) invoked right before each
    optimizer.step() -- used by finetune.py to apply LR warmup to the encoder
    param group without needing a torch LR scheduler tied to per-epoch calls."""
    if train:
        model.train()  # VistaClassifier.train() override keeps self.encoder.eval()
    else:
        model.eval()

    # frozen stages must always be in eval mode; VistaClassifier.train() override
    # guarantees this whether fully frozen (linear probe) or partially unfrozen
    # (fine-tuning some stages) -- only checks encoder-level eval() for full-freeze,
    # since a partially-unfrozen encoder legitimately has some sub-stages in train()
    if getattr(model, "freeze_encoder", True):
        assert model.encoder.training is False, "frozen encoder must always be in eval mode"
    else:
        finetune_stages = getattr(model, "finetune_stage_indices", None) or []
        for i, level in enumerate(model.encoder.layers):
            expected_training = train and (i in finetune_stages)
            assert level.training == expected_training, (
                f"encoder.layers[{i}].training={level.training}, expected {expected_training}"
            )

    total_loss = 0.0
    n_batches = 0
    all_y, all_p = [], []

    if train:
        optimizer.zero_grad(set_to_none=True)

    n = len(loader)
    desc = f"epoch {epoch}/{epochs} [{'train' if train else 'val'}]"
    pbar = tqdm(enumerate(loader), total=n, desc=desc, file=sys.stdout, leave=False)
    for i, (x, y, _paths) in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

        if train:
            scaled_loss = loss / accum_steps
            if amp and device == "cuda":
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_accum_boundary = ((i + 1) % accum_steps == 0) or ((i + 1) == n)
            if is_accum_boundary:
                # sanity: FROZEN encoder params (requires_grad=False) must never
                # accumulate a gradient. Params that are legitimately unfrozen for
                # fine-tuning (requires_grad=True) SHOULD have gradients here --
                # this is not a blanket "zero encoder gradients" check, since that
                # would be wrong whenever any encoder stage is being fine-tuned.
                leaked_grads = [
                    n_ for n_, p in model.encoder.named_parameters()
                    if (not p.requires_grad) and (p.grad is not None)
                ]
                assert not leaked_grads, (
                    f"frozen encoder params received gradients -- freeze broken: {leaked_grads[:5]}"
                )

                if step_counter is not None:
                    step_counter[0] += 1
                if on_optimizer_step is not None:
                    on_optimizer_step(step_counter[0] if step_counter is not None else None)

                if amp and device == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        n_batches += 1
        all_y.extend(y.detach().cpu().tolist())
        all_p.extend(torch.sigmoid(logits.detach().float()).cpu().tolist())

        if any(np.isnan(v) for v in [loss.item()]):
            raise RuntimeError(f"NaN loss at batch {i}")

        pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss/n_batches:.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(all_y, all_p)
    metrics["loss"] = avg_loss
    return metrics


def smoke_test(args):
    """Run a handful of accumulation cycles on real data and verify:
    no NaNs, no unexpected encoder grads, memory stays stable across steps."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== SMOKE TEST === device={device}")

    seeds = set_all_seeds(args.seed)
    splits, split_patients = patient_level_split(seed=args.split_seed)
    loaders = make_dataloaders(
        {"train": splits["train"][:args.accum_steps * 3 + 1]},  # just enough for 3 accumulation cycles
        batch_size=args.batch_size, patch_size=(args.patch_size,) * 3,
    )
    loader = loaders["train"]

    model = VistaClassifier(freeze_encoder=True).to(device)
    optimizer = build_optimizer(model, args.lr)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(args.amp and device == "cuda"))
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    assert model.encoder.training is False, "encoder must be in eval() after model.train()"
    print("Confirmed: model.train() called, model.encoder.training == False (frozen encoder stays in eval)")

    optimizer.zero_grad(set_to_none=True)
    peak_mems = []
    n = len(loader)
    print(f"Running {n} batches ({n // args.accum_steps} full accumulation cycles of {args.accum_steps})")

    for i, (x, y, paths) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        assert not torch.isnan(loss), f"NaN loss at batch {i}"
        assert not torch.isinf(loss), f"Inf loss at batch {i}"

        scaled_loss = loss / args.accum_steps
        if args.amp and device == "cuda":
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        is_boundary = ((i + 1) % args.accum_steps == 0) or ((i + 1) == n)
        if is_boundary:
            enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
            assert not enc_grads, f"encoder received gradients at batch {i} -- freeze is broken!"
            head_grads = [p.grad for p in model.head.parameters() if p.grad is not None]
            assert head_grads, f"head received NO gradients at batch {i} -- backward is broken!"
            for p in model.head.parameters():
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any(), f"NaN gradient in head at batch {i}"

            if args.amp and device == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            print(f"  batch {i+1}/{n}: accumulation boundary -> optimizer.step() "
                  f"(loss={loss.item():.4f}, {len(head_grads)} head tensors updated, 0 encoder grads)")

        if device == "cuda":
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            peak_mems.append(peak_mb)
            print(f"  batch {i+1}/{n}: loss={loss.item():.4f}, file={os.path.basename(paths[0])}, "
                  f"peak_mem={peak_mb:.1f} MB")

    if device == "cuda" and len(peak_mems) > 1:
        mem_range = max(peak_mems) - min(peak_mems)
        print(f"\nPeak memory across batches: min={min(peak_mems):.1f} MB, max={max(peak_mems):.1f} MB, "
              f"range={mem_range:.1f} MB")
        # a growing trend (not just per-batch variance from different volume sizes pre-crop)
        # would indicate a leak; here every batch is resized to a fixed patch_size so peak
        # memory per step should be flat modulo AMP/caching-allocator noise
        if mem_range > 0.25 * max(peak_mems):
            print("WARNING: peak memory varies by >25% across identical-shape batches -- possible leak")
        else:
            print("Memory stable across steps: OK")

    print("\n=== SMOKE TEST PASSED ===")
    print(f"Seeds used: {seeds}")
    return True


def run_linear_probe(splits, run_dir, args, progress_cb=None):
    """Core linear-probe training loop, parameterized by an explicit `splits`
    dict (train/val/test filepaths) instead of always deriving one internally
    -- lets callers (CLI main_train, or a cross-validation orchestrator) supply
    fold-specific splits. Uses UNGATED patience-based early stopping (stops
    as soon as `args.patience` epochs pass with no val-metric improvement,
    with no min_epochs floor) rather than always running a fixed epoch count.

    progress_cb(epoch, val_metrics): optional callback invoked after every
    epoch, for an external orchestrator to update a crash-diagnosis log.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    air_mask_threshold = getattr(args, "air_mask_threshold", None)
    air_mask_classifier_path = getattr(args, "air_mask_classifier_path", None)
    boundary_distance_channel = getattr(args, "boundary_distance_channel", False)
    in_channels = 2 if (air_mask_threshold is not None or boundary_distance_channel) else 1

    config = vars(args).copy()
    config["device"] = device
    config["gpu_name"] = torch.cuda.get_device_name(0) if device == "cuda" else None
    config["selection_metric"] = args.selection_metric
    config["air_mask_threshold"] = air_mask_threshold
    config["air_mask_classifier_path"] = air_mask_classifier_path
    config["boundary_distance_channel"] = boundary_distance_channel
    config["in_channels"] = in_channels
    with open(os.path.join(run_dir, "split.json"), "w") as f:
        json.dump({k: v for k, v in splits.items()}, f, indent=2)

    loaders = make_dataloaders(
        splits, batch_size=args.batch_size, patch_size=(args.patch_size,) * 3,
        num_workers=args.num_workers, augment_train=getattr(args, "augment", False),
        air_mask_threshold=air_mask_threshold, air_mask_classifier_path=air_mask_classifier_path,
        boundary_distance_channel=boundary_distance_channel,
    )
    print({k: len(v) for k, v in splits.items()})
    if air_mask_threshold is not None:
        refined_note = " (refined by stage-2 component classifier)" if air_mask_classifier_path else ""
        print(f"Air-mask channel ENABLED: threshold={air_mask_threshold} HU, in_channels=2{refined_note}")
    if boundary_distance_channel:
        print("Boundary-distance channel ENABLED: in_channels=2")

    # class imbalance in the train split: weight the positive class by neg/pos
    # ratio so BCEWithLogitsLoss doesn't let the model collapse to predicting
    # the majority (negative) class -- computed from the ACTUAL train split
    # counts (patient-level splitting can shift the ratio from the full-dataset
    # 145/218), not the full-dataset ratio.
    train_df = pd.read_csv(os.path.join(ROOT, "scripts", "audit_results.csv"))
    train_sub = train_df[train_df["filepath"].isin(splits["train"])]
    n_pos = (train_sub["label"] == "positive").sum()
    n_neg = (train_sub["label"] == "negative").sum()
    pos_weight_value = n_neg / n_pos
    print(f"Train split class balance: {n_pos} positive, {n_neg} negative "
          f"-> pos_weight={pos_weight_value:.4f}")
    config["train_n_pos"] = int(n_pos)
    config["train_n_neg"] = int(n_neg)
    config["pos_weight"] = float(pos_weight_value)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {os.path.join(run_dir, 'config.json')}")

    model = VistaClassifier(freeze_encoder=True, hidden_dim=args.hidden_dim, dropout=args.dropout,
                             in_channels=in_channels).to(device)
    optimizer = build_optimizer(model, args.lr)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(args.amp and device == "cuda"))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,split,loss,auroc,auprc,sensitivity,specificity,accuracy,elapsed_s\n")

    best_score = -float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    assert args.selection_metric in ("auprc", "auroc", "composite")

    patience = getattr(args, "patience", None)
    print(f"\nStarting linear-probe training: max_epochs={args.epochs} (ceiling), "
          f"patience={patience if patience else 'disabled (fixed epoch count)'} "
          f"(UNGATED -- applies from epoch 1, no min_epochs floor), "
          f"{len(loaders['train'])} train batches/epoch, {len(loaders['val'])} val batches/epoch\n")

    final_epoch = 0
    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        t0 = time.time()
        train_metrics = run_epoch(model, loaders["train"], optimizer, scaler, criterion,
                                   device, args.accum_steps, train=True, amp=args.amp,
                                   epoch=epoch, epochs=args.epochs)
        val_metrics = run_epoch(model, loaders["val"], optimizer, scaler, criterion,
                                 device, args.accum_steps, train=False, amp=args.amp,
                                 epoch=epoch, epochs=args.epochs)
        elapsed = time.time() - t0

        print(f"[epoch {epoch}/{args.epochs}] "
              f"train_loss={train_metrics['loss']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_auroc={val_metrics['auroc']:.4f} "
              f"val_auprc={val_metrics['auprc']:.4f} val_sens={val_metrics['sensitivity']:.4f} "
              f"val_spec={val_metrics['specificity']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"({elapsed:.1f}s)")

        with open(log_path, "a") as f:
            f.write(f"{epoch},train,{train_metrics['loss']:.6f},,,,,,{elapsed:.1f}\n")
            f.write(f"{epoch},val,{val_metrics['loss']:.6f},{val_metrics['auroc']:.6f},"
                    f"{val_metrics['auprc']:.6f},{val_metrics['sensitivity']:.6f},"
                    f"{val_metrics['specificity']:.6f},{val_metrics['accuracy']:.6f},{elapsed:.1f}\n")

        score = val_metrics[args.selection_metric]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics, "config": config,
                "selection_metric": args.selection_metric, "selection_score": score,
            }, os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> new best ({args.selection_metric}={score:.4f}), saved checkpoints/best.pt")
        else:
            epochs_since_improvement += 1

        if progress_cb is not None:
            progress_cb(epoch, val_metrics)

        if patience and epochs_since_improvement >= patience:
            print(f"\nEarly stopping: no val {args.selection_metric} improvement for "
                  f"{patience} epochs (ungated -- no min_epochs requirement). "
                  f"Best epoch={best_epoch}, score={best_score:.4f}.")
            break

    torch.save({
        "epoch": final_epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "config": config,
    }, os.path.join(ckpt_dir, "last.pt"))

    print(f"\nTraining complete. Best {args.selection_metric}={best_score:.4f} at epoch {best_epoch}.")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Log: {log_path}")
    return {"best_epoch": best_epoch, "best_score": best_score, "ckpt_dir": ckpt_dir}


def main_train(args):
    seeds = set_all_seeds(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    config = vars(args).copy()
    config["seeds"] = seeds
    config["timestamp"] = timestamp
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2))

    splits, split_patients = patient_level_split(seed=args.split_seed)
    run_linear_probe(splits, run_dir, args)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-size", type=int, default=224, help="cubic patch side length")
    p.add_argument("--batch-size", type=int, default=1, help="physical batch size per forward pass")
    p.add_argument("--accum-steps", type=int, default=4, help="gradient accumulation steps (effective batch = batch_size * accum_steps)")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--epochs", type=int, default=30, help="max epochs (ceiling)")
    p.add_argument("--patience", type=int, default=None,
                    help="if set, UNGATED early stopping: stop after this many epochs with no val-metric "
                         "improvement, starting from epoch 1 (no min_epochs floor). Default None = disabled, "
                         "always runs the fixed --epochs count (backward-compatible CLI default).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "runs"))
    p.add_argument("--selection-metric", type=str, default="composite",
                    choices=["auprc", "auroc", "composite"],
                    help="'composite' (default) blends auroc/auprc/sensitivity/specificity to avoid "
                         "picking a degenerate near-one-class checkpoint that AUPRC alone can favor -- "
                         "see COMPOSITE_WEIGHT_* constants above compute_metrics()")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--augment", action="store_true", default=False,
                    help="apply train-only 3D augmentation (flip/rotate/intensity jitter, see augment.py)")
    p.add_argument("--air-mask-threshold", type=float, default=None,
                    help="if set, add a 2nd input channel: 1.0 where raw HU < threshold else 0.0 "
                         "(e.g. -600 for air). Default None = original 1-channel (intensity only).")
    p.add_argument("--air-mask-classifier-path", type=str, default=None,
                    help="optional stage-2 refinement: path to a pickled component classifier "
                         "(from train_component_classifier.py) that reweights the air-mask channel "
                         "by per-component probability instead of a raw threshold. Only meaningful "
                         "when --air-mask-threshold is also set.")
    p.add_argument("--boundary-distance-channel", action="store_true", default=False,
                    help="alternative aux channel to --air-mask-threshold: a smooth, unthresholded "
                         "physical distance-to-body-boundary field (mm, normalized to [0,1]) instead "
                         "of a binary/refined air mask. Mutually exclusive with --air-mask-threshold.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        smoke_test(args)
    else:
        main_train(args)
