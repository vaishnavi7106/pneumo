"""
Step 10: partial fine-tuning starting from the best linear-probe checkpoint.

Unfreezes only the deepest 1-2 SegResEncoder stages (stage 4 alone, or
stage 3+4) instead of the full 175M-param encoder, to control overfitting
risk and training time. Differential learning rates: a low LR for the
newly-unfrozen encoder stages, a higher LR for the classification head.

IMPORTANT param-count caveat (see stage breakdown below): the "deepest 1-2
stages" framing undersells how much capacity this actually unfreezes --
stage 4 alone is 127.4M of the encoder's 175.0M params (72.8%), and stage
3+4 together is 167.3M (95.6%). Unfreezing 2 stages is barely different from
unfreezing the whole encoder in terms of trainable capacity, even though
it's a small fraction of the *stage count*. Default here is stage 4 ONLY.

Reuses run_epoch/compute_metrics/set_all_seeds from train.py so both scripts
share identical metric computation and accumulation logic.
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import make_dataloaders  # noqa: E402
from model import VistaClassifier  # noqa: E402
from split import patient_level_split  # noqa: E402
from train import build_optimizer, compute_metrics, run_epoch, set_all_seeds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_latest_best_checkpoint(runs_dir: str) -> str:
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "run_*")))
    for run_dir in reversed(run_dirs):
        ckpt = os.path.join(run_dir, "checkpoints", "best.pt")
        if os.path.exists(ckpt):
            return ckpt
    raise FileNotFoundError(f"No checkpoints/best.pt found under any run_* in {runs_dir}")


def report_stage_param_counts(model: VistaClassifier):
    print("\nEncoder stage parameter counts (for reference on unfreeze scope):")
    total = sum(p.numel() for p in model.encoder.parameters())
    for i, level in enumerate(model.encoder.layers):
        n = sum(p.numel() for p in level.parameters())
        print(f"  stage {i}: {n:,} params ({100*n/total:.1f}% of encoder)")
    print(f"  encoder total: {total:,} params\n")


def build_finetune_optimizer(model: VistaClassifier, encoder_lr: float, head_lr: float):
    encoder_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    other = [n for n, p in model.named_parameters()
             if p.requires_grad and not n.startswith("encoder.") and not n.startswith("head.")]
    assert not other, f"unexpected trainable params outside encoder/head: {other}"

    n_enc = sum(p.numel() for p in encoder_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"Optimizer param groups: encoder={len(encoder_params)} tensors/{n_enc:,} params @ lr={encoder_lr}, "
          f"head={len(head_params)} tensors/{n_head:,} params @ lr={head_lr}")

    return torch.optim.Adam([
        {"params": encoder_params, "lr": encoder_lr, "name": "encoder"},
        {"params": head_params, "lr": head_lr, "name": "head"},
    ])


def load_from_checkpoint(checkpoint_path: str, unfreeze_stages, device: str,
                          expected_air_mask_threshold=None, expected_air_mask_classifier_path=None,
                          expected_boundary_distance_channel=False
                          ) -> VistaClassifier:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    src_config = ckpt.get("config", {})
    hidden_dim = src_config.get("hidden_dim", 128)
    dropout = src_config.get("dropout", 0.3)
    in_channels = src_config.get("in_channels", 1)

    # catch an accidental mismatch (e.g. fine-tuning with --air-mask-threshold
    # set/unset differently than the source linear-probe checkpoint was
    # trained with) before it silently trains a randomly-shaped conv_init
    src_air_mask_threshold = src_config.get("air_mask_threshold")
    assert src_air_mask_threshold == expected_air_mask_threshold, (
        f"air_mask_threshold mismatch: source checkpoint was trained with "
        f"{src_air_mask_threshold!r}, but this run was asked for {expected_air_mask_threshold!r}"
    )
    # same idea for the stage-2 refinement -- doesn't change in_channels (so
    # wouldn't crash on a shape mismatch) but would silently fine-tune on
    # differently-computed mask content than the linear probe learned from
    src_air_mask_classifier_path = src_config.get("air_mask_classifier_path")
    assert src_air_mask_classifier_path == expected_air_mask_classifier_path, (
        f"air_mask_classifier_path mismatch: source checkpoint was trained with "
        f"{src_air_mask_classifier_path!r}, but this run was asked for {expected_air_mask_classifier_path!r}"
    )
    # same mismatch guard for the boundary-distance channel alternative
    src_boundary_distance_channel = src_config.get("boundary_distance_channel", False)
    assert src_boundary_distance_channel == expected_boundary_distance_channel, (
        f"boundary_distance_channel mismatch: source checkpoint was trained with "
        f"{src_boundary_distance_channel!r}, but this run was asked for {expected_boundary_distance_channel!r}"
    )

    model = VistaClassifier(freeze_encoder=True, hidden_dim=hidden_dim, dropout=dropout,
                             in_channels=in_channels)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  source epoch={ckpt.get('epoch')}, "
          f"selection_metric={ckpt.get('selection_metric')}, score={ckpt.get('selection_score')}")
    print(f"  source val_metrics={ckpt.get('val_metrics')}")

    report_stage_param_counts(model)
    model.set_encoder_finetune_stages(unfreeze_stages)
    model.to(device)
    return model, ckpt


def run_finetune(splits, checkpoint_path, run_dir, args, progress_cb=None):
    """Core fine-tuning loop, parameterized by an explicit `splits` dict and
    `checkpoint_path` instead of always deriving them internally -- lets
    callers (CLI main_finetune, or a cross-validation orchestrator) supply
    fold-specific splits and a fold-specific source linear-probe checkpoint.

    progress_cb(epoch, val_metrics): optional callback invoked after every
    epoch, for an external orchestrator to update a crash-diagnosis log.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    unfreeze_stages = {1: [4], 2: [3, 4], 3: [3]}[args.unfreeze_stages]

    with open(os.path.join(run_dir, "split.json"), "w") as f:
        json.dump({k: v for k, v in splits.items()}, f, indent=2)

    air_mask_threshold = getattr(args, "air_mask_threshold", None)
    air_mask_classifier_path = getattr(args, "air_mask_classifier_path", None)
    boundary_distance_channel = getattr(args, "boundary_distance_channel", False)
    loaders = make_dataloaders(
        splits, batch_size=args.batch_size, patch_size=(args.patch_size,) * 3,
        num_workers=args.num_workers, augment_train=getattr(args, "augment", False),
        air_mask_threshold=air_mask_threshold, air_mask_classifier_path=air_mask_classifier_path,
        boundary_distance_channel=boundary_distance_channel,
    )
    print({k: len(v) for k, v in splits.items()})

    train_df = pd.read_csv(os.path.join(ROOT, "scripts", "audit_results.csv"))
    train_sub = train_df[train_df["filepath"].isin(splits["train"])]
    n_pos = (train_sub["label"] == "positive").sum()
    n_neg = (train_sub["label"] == "negative").sum()
    pos_weight_value = n_neg / n_pos
    print(f"Train split class balance: {n_pos} positive, {n_neg} negative -> pos_weight={pos_weight_value:.4f}")

    model, src_ckpt = load_from_checkpoint(checkpoint_path, unfreeze_stages, device,
                                            expected_air_mask_threshold=air_mask_threshold,
                                            expected_air_mask_classifier_path=air_mask_classifier_path,
                                            expected_boundary_distance_channel=boundary_distance_channel)
    optimizer = build_finetune_optimizer(model, args.encoder_lr, args.head_lr)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(args.amp and device == "cuda"))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # LR warmup for the newly-unfrozen encoder stage(s): linear ramp from
    # warmup_start_lr (default 0) up to the target encoder_lr over the first
    # warmup_epochs worth of optimizer steps. The head's LR is NOT warmed up --
    # the head was already trained during the linear probe, so its optimizer
    # state and gradient scale are already well-conditioned; warmup exists
    # specifically to protect the freshly-unfrozen 127M-param stage 4 from
    # taking full-strength (lr=1e-5) steps before its running Adam moment
    # estimates have had a chance to stabilize, which is what produced the
    # sens/spec swings observed without warmup (0.86/0.15 -> 0.43/0.67 ->
    # 0.095/1.0 across 3 epochs).
    steps_per_epoch = -(-len(loaders["train"]) // args.accum_steps)  # ceil div
    warmup_steps = max(1, int(round(args.warmup_epochs * steps_per_epoch)))
    print(f"LR warmup: encoder lr {args.warmup_start_lr:.2e} -> {args.encoder_lr:.2e} "
          f"linearly over {warmup_steps} optimizer steps ({args.warmup_epochs} epochs "
          f"x {steps_per_epoch} steps/epoch). Head lr starts at {args.head_lr:.2e} "
          f"(already trained during linear probe, no warmup -- decay only, see below).")

    # LR decay AFTER warmup: cosine decay from each group's peak LR down to
    # lr_min_frac*peak over the epochs from warmup_epochs to max_epochs. This
    # is composed with warmup (not a separate phase gated on a fixed epoch):
    # decay_progress is clamped to 0 while still inside the warmup window, so
    # during warmup the encoder ramps 0->peak while decay_mult==1, and only
    # after warmup ends does the cosine decay start pulling both groups down.
    # Added because the linear-probe resume run showed the head oscillating
    # late in training (epoch 39->40 AUPRC swing 0.747->0.532) with a constant
    # LR and no decay -- decay damps exactly this kind of late-training noise.
    print(f"LR decay: cosine, peak -> {args.lr_min_frac:.0%} of peak, over epochs "
          f"[{args.warmup_epochs}, {args.lr_decay_epochs}] (applies to both encoder and head; "
          f"decoupled from max_epochs={args.max_epochs} -- reaches floor well before the ceiling "
          f"so later epochs, if reached, settle instead of continuing to take near-peak-sized steps)")

    ENCODER_GROUP_IDX, HEAD_GROUP_IDX = 0, 1
    assert optimizer.param_groups[ENCODER_GROUP_IDX].get("name") == "encoder"
    assert optimizer.param_groups[HEAD_GROUP_IDX].get("name") == "head"

    import math

    def decay_multiplier(global_step):
        epoch_frac = global_step / steps_per_epoch
        denom = max(1e-9, args.lr_decay_epochs - args.warmup_epochs)
        progress = min(1.0, max(0.0, (epoch_frac - args.warmup_epochs) / denom))
        cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return args.lr_min_frac + (1 - args.lr_min_frac) * cos_factor

    def apply_lr_schedule(global_step):
        warmup_frac = min(1.0, global_step / warmup_steps)
        encoder_warmed_up_lr = args.warmup_start_lr + (args.encoder_lr - args.warmup_start_lr) * warmup_frac
        mult = decay_multiplier(global_step)
        optimizer.param_groups[ENCODER_GROUP_IDX]["lr"] = encoder_warmed_up_lr * mult
        optimizer.param_groups[HEAD_GROUP_IDX]["lr"] = args.head_lr * mult

    apply_lr_schedule(0)  # start at warmup_start_lr for encoder, head_lr*1.0 for head
    step_counter = [0]

    config = vars(args).copy()
    config.update({
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "source_checkpoint": checkpoint_path,
        "source_epoch": src_ckpt.get("epoch"), "source_val_metrics": src_ckpt.get("val_metrics"),
        "unfreeze_stage_indices": unfreeze_stages, "in_channels": model.in_channels,
        "train_n_pos": int(n_pos), "train_n_neg": int(n_neg), "pos_weight": float(pos_weight_value),
        "steps_per_epoch": steps_per_epoch, "warmup_steps": warmup_steps,
    })
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {os.path.join(run_dir, 'config.json')}")

    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,split,loss,auroc,auprc,sensitivity,specificity,accuracy,elapsed_s,encoder_lr,head_lr\n")

    best_score = -float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    epoch_times = []

    print(f"\nStarting fine-tuning: max_epochs={args.max_epochs} (ceiling), "
          f"patience={args.patience} epochs (UNGATED -- applies from epoch 1, no min_epochs floor), "
          f"{len(loaders['train'])} train batches/epoch, {len(loaders['val'])} val batches/epoch\n")

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, loaders["train"], optimizer, scaler, criterion,
                                   device, args.accum_steps, train=True, amp=args.amp,
                                   epoch=epoch, epochs=args.max_epochs,
                                   step_counter=step_counter, on_optimizer_step=apply_lr_schedule)
        val_metrics = run_epoch(model, loaders["val"], optimizer, scaler, criterion,
                                 device, args.accum_steps, train=False, amp=args.amp,
                                 epoch=epoch, epochs=args.max_epochs)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        current_encoder_lr = optimizer.param_groups[ENCODER_GROUP_IDX]["lr"]
        current_head_lr = optimizer.param_groups[HEAD_GROUP_IDX]["lr"]
        print(f"[epoch {epoch}/{args.max_epochs}] encoder_lr={current_encoder_lr:.3e} "
              f"head_lr={current_head_lr:.3e} "
              f"train_loss={train_metrics['loss']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_auroc={val_metrics['auroc']:.4f} "
              f"val_auprc={val_metrics['auprc']:.4f} val_sens={val_metrics['sensitivity']:.4f} "
              f"val_spec={val_metrics['specificity']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"({elapsed:.1f}s)")

        with open(log_path, "a") as f:
            f.write(f"{epoch},train,{train_metrics['loss']:.6f},,,,,,{elapsed:.1f},"
                    f"{current_encoder_lr:.6e},{current_head_lr:.6e}\n")
            f.write(f"{epoch},val,{val_metrics['loss']:.6f},{val_metrics['auroc']:.6f},"
                    f"{val_metrics['auprc']:.6f},{val_metrics['sensitivity']:.6f},"
                    f"{val_metrics['specificity']:.6f},{val_metrics['accuracy']:.6f},{elapsed:.1f},"
                    f"{current_encoder_lr:.6e},{current_head_lr:.6e}\n")

        # per-epoch checkpoint for full-trajectory visibility, independent of best-tracking
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics, "config": config,
        }, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

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
                "unfreeze_stage_indices": unfreeze_stages,
            }, os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> new best ({args.selection_metric}={score:.4f}), saved checkpoints/best.pt")
        else:
            epochs_since_improvement += 1

        if progress_cb is not None:
            progress_cb(epoch, val_metrics)

        # UNGATED early stopping: patience applies from epoch 1, no min_epochs floor.
        # This directly replaces the previous run's structural flaw (patience gated
        # behind min_epochs=50), which forced training to continue ~25 epochs past
        # its epoch-5 optimum before any stop condition could even be checked.
        if epochs_since_improvement >= args.patience:
            print(f"\nEarly stopping: no val {args.selection_metric} improvement for "
                  f"{args.patience} epochs (ungated -- no min_epochs requirement). "
                  f"Best epoch={best_epoch}, score={best_score:.4f}.")
            break

        if args.time_probe_epochs and epoch >= args.time_probe_epochs:
            avg_epoch_s = sum(epoch_times) / len(epoch_times)
            print(f"\n=== TIME PROBE: stopping after {epoch} epoch(s) as requested ===")
            print(f"Average epoch time: {avg_epoch_s:.1f}s")
            for target_epochs in [args.patience, args.max_epochs]:
                total_s = avg_epoch_s * target_epochs
                print(f"Estimated total for {target_epochs} epochs: {total_s/60:.1f} min "
                      f"({total_s/3600:.2f} hours)")
            return {"best_epoch": best_epoch, "best_score": best_score, "ckpt_dir": ckpt_dir}

    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "config": config,
    }, os.path.join(ckpt_dir, "last.pt"))

    print(f"\nFine-tuning complete. Best {args.selection_metric}={best_score:.4f} at epoch {best_epoch}.")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Log: {log_path}")
    return {"best_epoch": best_epoch, "best_score": best_score, "ckpt_dir": ckpt_dir}


def main_finetune(args):
    seeds = set_all_seeds(args.seed)

    checkpoint_path = args.checkpoint or find_latest_best_checkpoint(args.runs_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"finetune_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    splits, split_patients = patient_level_split(seed=args.split_seed)
    run_finetune(splits, checkpoint_path, run_dir, args)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                    help="path to linear-probe checkpoints/best.pt (default: auto-find latest under --runs-dir)")
    p.add_argument("--runs-dir", type=str, default=os.path.join(ROOT, "runs"))
    p.add_argument("--unfreeze-stages", type=int, default=1, choices=[1, 2, 3],
                    help="1 = unfreeze stage 4 only (127.4M params, 72.8%% of encoder); "
                         "2 = unfreeze stage 3+4 (167.3M params, 95.6%% of encoder); "
                         "3 = unfreeze stage 3 only (39.8M params, 22.8%% of encoder)")
    p.add_argument("--patch-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=1,
                    help="physical batch size; default 1 because batch_size=4 OOMs and batch_size=2 "
                         "leaves no safety margin (11.8GB reserved of 12GB) when fine-tuning stage 4 -- "
                         "use with --accum-steps 4 for an effective batch of 4")
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--max-epochs", type=int, default=50, help="hard ceiling on epochs")
    p.add_argument("--patience", type=int, default=15,
                    help="early-stop after this many epochs with no val-metric improvement, "
                         "UNGATED -- applies starting from epoch 1, no min_epochs floor")
    p.add_argument("--encoder-lr", type=float, default=1e-5,
                    help="target encoder LR after warmup completes")
    p.add_argument("--head-lr", type=float, default=1e-3,
                    help="head LR, constant throughout, no warmup (head already trained in linear probe)")
    p.add_argument("--warmup-epochs", type=float, default=2,
                    help="linearly ramp encoder LR from --warmup-start-lr to --encoder-lr over this "
                         "many epochs' worth of optimizer steps")
    p.add_argument("--warmup-start-lr", type=float, default=0.0)
    p.add_argument("--lr-decay-epochs", type=float, default=22,
                    help="epoch at which cosine decay reaches its floor -- decoupled from --max-epochs "
                         "so the LR is fully settled well before a long run's later epochs, instead of "
                         "still taking near-peak-sized steps deep into training")
    p.add_argument("--lr-min-frac", type=float, default=0.1,
                    help="cosine LR decay floor as a fraction of each group's peak LR, reached at "
                         "--lr-decay-epochs; decay applies to both encoder and head after warmup ends")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "runs"))
    p.add_argument("--selection-metric", type=str, default="composite",
                    choices=["auprc", "auroc", "composite"],
                    help="'composite' (default) blends auroc/auprc/sensitivity/specificity to avoid "
                         "picking a degenerate near-one-class checkpoint that AUPRC alone can favor -- "
                         "see COMPOSITE_WEIGHT_* constants in train.py's compute_metrics()")
    p.add_argument("--time-probe-epochs", type=int, default=0,
                    help="if >0, run only this many epochs then report per-epoch timing and ETA, then exit")
    p.add_argument("--augment", action="store_true", default=False,
                    help="apply train-only 3D augmentation (flip/rotate/intensity jitter, see augment.py)")
    p.add_argument("--air-mask-threshold", type=float, default=None,
                    help="must match the source linear-probe checkpoint's setting -- if it was trained "
                         "with a 2nd air-mask channel, this run must use the same threshold")
    p.add_argument("--air-mask-classifier-path", type=str, default=None,
                    help="must match the source linear-probe checkpoint's setting")
    p.add_argument("--boundary-distance-channel", action="store_true", default=False,
                    help="must match the source linear-probe checkpoint's setting")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main_finetune(args)
