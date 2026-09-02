"""
Same 224^3 memory probe as find_batch_size_224.py, but with AMP (autocast +
GradScaler) to see whether that makes 224^3 training practical on the 3060.

Everything else identical for a clean comparison: same sample volume, same
preprocessing pipeline, same VistaClassifier (encoder still fully frozen --
AMP does not unfreeze anything, it just runs the frozen encoder's forward
pass in fp16 to cut activation memory), same real training step (forward ->
loss -> backward -> optimizer.step()), same allocator cap to avoid Windows
sysmem-fallback masking the true VRAM boundary.

RTX 3060 is Ampere -- both fp16 and bf16 are hardware-supported. Using fp16
per instructions (not assuming bf16 is preferable), which requires
GradScaler for stable gradients since fp16 has a much narrower exponent
range than bf16/fp32.
"""
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.transforms import Orientation, Spacing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import HU_A_MAX, HU_A_MIN, RESAMPLE_SPACING, VistaClassifier  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE_224 = (224, 224, 224)


def preprocess(filepath: str, patch_size) -> torch.Tensor:
    img = nib.load(filepath)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine

    data = data[None]
    data = MetaTensor(torch.as_tensor(data, dtype=torch.float32), affine=torch.as_tensor(affine, dtype=torch.float64))
    data = Orientation(axcodes="RAS")(data)
    data = Spacing(pixdim=RESAMPLE_SPACING, mode="bilinear")(data)
    data = torch.as_tensor(np.asarray(data), dtype=torch.float32)

    data = torch.clamp(data, HU_A_MIN, HU_A_MAX)
    data = (data - HU_A_MIN) / (HU_A_MAX - HU_A_MIN)

    data = data.unsqueeze(0)
    data = F.interpolate(data, size=patch_size, mode="trilinear", align_corners=False)
    return data.squeeze(0)


def build_batch(sample: torch.Tensor, batch_size: int) -> torch.Tensor:
    return sample.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).contiguous()


def try_batch_size_amp(model, optimizer, scaler, criterion, sample: torch.Tensor, batch_size: int):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = build_batch(sample, batch_size).to(DEVICE)
    y = torch.randint(0, 2, (batch_size,), dtype=torch.float32, device=DEVICE)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        # NOTE: model.forward() already wraps the encoder call in torch.no_grad()
        # when freeze_encoder=True (see model.py) -- that no_grad still applies
        # here, autocast just changes the compute dtype of that no_grad forward,
        # it does not unfreeze or backprop through the encoder.
        logits = model(x)
        assert logits.requires_grad, "logits should require grad (head is trainable)"
        loss = criterion(logits, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / 1e6
    peak_reserved = torch.cuda.max_memory_reserved() / 1e6

    encoder_grad_count = sum(1 for p in model.encoder.parameters() if p.grad is not None)
    return peak_alloc, peak_reserved, loss.item(), encoder_grad_count


def get_gpu_memory_status():
    """Query nvidia-smi directly for ground truth: dedicated VRAM used/free."""
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    total, used, free = map(int, out.split(","))
    return total, used, free


def main():
    df = pd.read_csv(AUDIT_CSV)
    row = df[(df.extent_z_mm >= 385) & (df.extent_z_mm <= 685)].iloc[0]
    print(f"Using sample volume: {row.filename} (extent_z={row.extent_z_mm:.0f}mm)")
    print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'n/a'}")

    if DEVICE == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory
        frac = (11 * 1024**3) / total_mem
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
        print(f"Capped allocator to ~{frac*total_mem/1e9:.1f}GB of {total_mem/1e9:.1f}GB total")

    sample_224 = preprocess(row.filepath, PATCH_SIZE_224)
    print(f"224^3 sample shape: {tuple(sample_224.shape)}")

    model = VistaClassifier(freeze_encoder=True).to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    print(f"\nAMP dtype: float16 (RTX 3060 is Ampere, hardware-supports both fp16/bf16; "
          f"using fp16 per instructions, with GradScaler for stability)")

    results = {}
    for bs in [1, 2]:
        print(f"\n=== AMP, batch_size={bs} ===")
        torch.cuda.empty_cache()  # release cached blocks from any previous run before measuring "before" state
        total_before, used_before, free_before = get_gpu_memory_status()
        try:
            peak_alloc, peak_reserved, loss_val, enc_grads = try_batch_size_amp(
                model, optimizer, scaler, criterion, sample_224, bs
            )
            total_after, used_after, free_after = get_gpu_memory_status()

            print(f"peak_alloc={peak_alloc:.1f} MB, peak_reserved={peak_reserved:.1f} MB, loss={loss_val:.4f}")
            print(f"encoder params with grad (should be 0, frozen): {enc_grads}")
            print(f"nvidia-smi memory.used: before={used_before}MiB -> after={used_after}MiB, "
                  f"free={free_after}MiB / {total_after}MiB total")

            exceeds_free = peak_reserved > free_before
            print(f"peak_reserved ({peak_reserved:.1f} MB) vs free VRAM before run ({free_before} MB): "
                  f"{'EXCEEDS -- likely spilled to shared system memory' if exceeds_free else 'fits within dedicated VRAM'}")

            results[bs] = {
                "status": "ok", "peak_alloc": peak_alloc, "peak_reserved": peak_reserved,
                "exceeds_free_vram": exceeds_free, "loss": loss_val,
            }

            if exceeds_free:
                print(f"NOTE: batch_size={bs} does not fit comfortably -- stopping sweep here.")
                break
        except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001
            print(f"OOM -- {str(e).splitlines()[0]}")
            torch.cuda.empty_cache()
            results[bs] = {"status": "oom"}
            break

    print("\n=== COMPARISON: FP32 vs AMP(fp16) at 224^3, batch_size=1 ===")
    print(f"{'metric':30s} {'FP32 (previous)':>20s} {'AMP fp16 (this run)':>20s}")
    prev_alloc, prev_reserved = 9452.0, 11530.1
    if 1 in results and results[1]["status"] == "ok":
        r = results[1]
        print(f"{'peak_alloc (MB)':30s} {prev_alloc:20.1f} {r['peak_alloc']:20.1f}")
        print(f"{'peak_reserved (MB)':30s} {prev_reserved:20.1f} {r['peak_reserved']:20.1f}")
        print(f"{'reduction in peak_reserved':30s} {'':20s} {100*(1-r['peak_reserved']/prev_reserved):19.1f}%")
    else:
        print("AMP batch_size=1 did not succeed -- see OOM above.")


if __name__ == "__main__":
    main()
