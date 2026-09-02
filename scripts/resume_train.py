"""
Resume the linear probe from a checkpoint for additional epochs, using the
same config/split/seed as the original run, to check whether val AUPRC keeps
climbing past where the original run stopped or has genuinely plateaued.
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import make_dataloaders  # noqa: E402
from model import VistaClassifier  # noqa: E402
from split import patient_level_split  # noqa: E402
from train import build_optimizer, run_epoch, set_all_seeds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--extra-epochs", type=int, default=20)
    p.add_argument("--selection-metric", type=str, default="auprc", choices=["auprc", "auroc"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    start_epoch = ckpt["epoch"]
    print(f"Resuming from {args.checkpoint} (epoch {start_epoch}, "
          f"{config['selection_metric']}={ckpt.get('selection_score')})")
    print(f"Using original config: patch_size={config['patch_size']}, batch_size={config['batch_size']}, "
          f"accum_steps={config['accum_steps']}, lr={config['lr']}, seed={config['seed']}, "
          f"split_seed={config['split_seed']}")

    set_all_seeds(config["seed"])

    src_run_dir = os.path.dirname(os.path.dirname(args.checkpoint))  # .../checkpoints/best.pt -> run dir
    run_dir = src_run_dir  # continue writing into the SAME run dir, not a new one
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_path = os.path.join(run_dir, "train_log.csv")
    assert os.path.exists(log_path), f"expected existing train_log.csv at {log_path}"

    splits, _ = patient_level_split(seed=config["split_seed"])
    loaders = make_dataloaders(
        splits, batch_size=config["batch_size"], patch_size=(config["patch_size"],) * 3,
        num_workers=config.get("num_workers", 0),
    )
    print({k: len(v) for k, v in splits.items()})

    model = VistaClassifier(freeze_encoder=True, hidden_dim=config["hidden_dim"],
                             dropout=config["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    optimizer = build_optimizer(model, config["lr"])
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Restored optimizer state (Adam moments) from checkpoint")
        except Exception as e:  # noqa: BLE001
            print(f"Could not restore optimizer state ({e}), starting fresh optimizer state")

    scaler = torch.amp.GradScaler(device="cuda", enabled=(config["amp"] and device == "cuda"))
    pos_weight = torch.tensor(config["pos_weight"], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score = ckpt.get("selection_score", -float("inf"))
    best_epoch = start_epoch
    print(f"Starting best {args.selection_metric}={best_score:.4f} (from epoch {start_epoch})")

    total_epochs = start_epoch + args.extra_epochs
    for epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, loaders["train"], optimizer, scaler, criterion,
                                   device, config["accum_steps"], train=True, amp=config["amp"],
                                   epoch=epoch, epochs=total_epochs)
        val_metrics = run_epoch(model, loaders["val"], optimizer, scaler, criterion,
                                 device, config["accum_steps"], train=False, amp=config["amp"],
                                 epoch=epoch, epochs=total_epochs)
        elapsed = time.time() - t0

        print(f"[epoch {epoch}/{total_epochs}] "
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
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics, "config": config,
                "selection_metric": args.selection_metric, "selection_score": score,
            }, os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> new best ({args.selection_metric}={score:.4f}), saved checkpoints/best.pt")

    torch.save({
        "epoch": total_epochs, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "config": config,
    }, os.path.join(ckpt_dir, "last.pt"))

    print(f"\nResume complete. Best {args.selection_metric}={best_score:.4f} at epoch {best_epoch} "
          f"(started resume at epoch {start_epoch} with {ckpt.get('selection_score'):.4f}).")


if __name__ == "__main__":
    main()
