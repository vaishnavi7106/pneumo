"""
Quick GPU memory dry-run for VistaClassifier at a given cubic (or non-cubic)
input size, WITHOUT touching real data/cache -- just random tensors through
a real forward (+ backward, for the fine-tune-shaped case) pass, so we can
catch an OOM in seconds instead of discovering it mid-training-run.

Checks both stages that matter memory-wise:
  1. Linear probe: encoder fully frozen, forward wrapped in no_grad (see
     VistaClassifier.forward) -- should be the cheaper of the two.
  2. Fine-tune: last encoder stage unfrozen (unfreeze_stages=1, the default
     first fine-tune stage in finetune.py) -- forward + backward, real
     gradients computed for the unfrozen stage + head.

Usage:
  python scripts/check_gpu_memory.py --size 384
  python scripts/check_gpu_memory.py --size 336 256 432
  python scripts/check_gpu_memory.py --size 384 --batch-size 2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from model import VistaClassifier


def fmt_gb(bytes_val):
    return f"{bytes_val / 1e9:.2f}GB"


def run_case(name, model, x, y, backward: bool):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        if backward:
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
        else:
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        print(f"[{name}] OK -- peak_allocated={fmt_gb(peak)}  peak_reserved={fmt_gb(reserved)}")
        return True
    except torch.cuda.OutOfMemoryError as e:
        print(f"[{name}] OOM: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, nargs="+", required=True,
                    help="one int for a cubic size (e.g. 384) or three ints D H W (e.g. 336 256 432)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--amp", action="store_true", default=True)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available -- this check requires a GPU.")
        return

    size = tuple(args.size) if len(args.size) == 3 else (args.size[0],) * 3
    print(f"GPU: {torch.cuda.get_device_name(0)}, total memory: "
          f"{fmt_gb(torch.cuda.get_device_properties(0).total_memory)}")
    print(f"Testing input size {size}, batch_size={args.batch_size}, in_channels={args.in_channels}, amp={args.amp}\n")

    x = torch.rand(args.batch_size, args.in_channels, *size, device="cuda")
    y = torch.randint(0, 2, (args.batch_size,), device="cuda").float()

    amp_ctx = torch.autocast("cuda", dtype=torch.float16) if args.amp else torch.cuda.amp.autocast(enabled=False)

    # --- case 1: linear probe (frozen encoder, no_grad) ---
    model = VistaClassifier(freeze_encoder=True, in_channels=args.in_channels).cuda()
    with amp_ctx:
        ok_lp = run_case("linear_probe (frozen, no_grad)", model, x, y, backward=False)
    del model
    torch.cuda.empty_cache()

    # --- case 2: fine-tune (last encoder stage unfrozen, forward+backward) ---
    model = VistaClassifier(freeze_encoder=True, in_channels=args.in_channels).cuda()
    model.set_encoder_finetune_stages([4])  # unfreeze_stages=1 equivalent in finetune.py
    model.train()
    with amp_ctx:
        ok_ft = run_case("fine_tune (stage 4 unfrozen, forward+backward)", model, x, y, backward=True)
    del model
    torch.cuda.empty_cache()

    print(f"\n=== SUMMARY (size={size}, batch_size={args.batch_size}) ===")
    print(f"linear probe: {'PASS' if ok_lp else 'OOM'}")
    print(f"fine-tune:    {'PASS' if ok_ft else 'OOM'}")


if __name__ == "__main__":
    main()
