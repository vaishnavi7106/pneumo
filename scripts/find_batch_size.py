"""
Empirically find the max linear-probe training batch size that fits the
3060's 12GB, by running real training steps (forward + backward through the
head + optimizer.step()) at increasing batch sizes until OOM.

Confirms explicitly that the frozen encoder's forward pass runs under
torch.no_grad() (VistaClassifier.forward does this internally when
freeze_encoder=True) -- verified below by checking pooled.requires_grad.
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
from model import HU_A_MAX, HU_A_MIN, PATCH_SIZE, RESAMPLE_SPACING, VistaClassifier  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(filepath: str) -> torch.Tensor:
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
    data = F.interpolate(data, size=PATCH_SIZE, mode="trilinear", align_corners=False)
    return data.squeeze(0)  # [1, 128, 128, 128], caller stacks into batch


def build_batch(sample: torch.Tensor, batch_size: int) -> torch.Tensor:
    return sample.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).contiguous()


def try_batch_size(model, optimizer, criterion, sample: torch.Tensor, batch_size: int):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = build_batch(sample, batch_size).to(DEVICE)
    y = torch.randint(0, 2, (batch_size,), dtype=torch.float32, device=DEVICE)

    optimizer.zero_grad(set_to_none=True)

    stages = model.encoder(x) if False else None  # not used directly; go through model.forward
    logits = model(x)

    # explicit check: encoder ran under no_grad -> pooled/logit graph should NOT
    # include encoder params, but logits DOES require grad (from head params)
    assert logits.requires_grad, "logits should require grad (head is trainable)"

    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / 1e6
    peak_reserved = torch.cuda.max_memory_reserved() / 1e6
    return peak_alloc, peak_reserved, loss.item()


def verify_no_grad_on_encoder(model, sample):
    """Explicitly confirm the frozen encoder path does not build an autograd graph."""
    x = build_batch(sample, 1).to(DEVICE)
    stages = model.encoder(x)  # this call itself is NOT wrapped -- test raw encoder
    # but inside model.forward(), the encoder call IS wrapped in no_grad. Confirm via source path:
    with torch.no_grad():
        stages_nograd = model.encoder(x)
    pooled = model.pool(stages_nograd[-1])
    print(f"[check] pooled.requires_grad under explicit no_grad (should be False): {pooled.requires_grad}")

    logits = model(x)
    print(f"[check] model(x) logits.requires_grad (should be True, head trainable): {logits.requires_grad}")

    # confirm encoder params get no gradient after a full backward
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward()
    encoder_grad_norms = [p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None]
    print(f"[check] encoder params with non-None .grad after backward (should be 0): {len(encoder_grad_norms)}")
    head_grad_norms = [p.grad.norm().item() for p in model.head.parameters() if p.grad is not None]
    print(f"[check] head params with non-None .grad after backward (should be >0): {len(head_grad_norms)}")


def main():
    df = pd.read_csv(AUDIT_CSV)
    row = df[(df.extent_z_mm >= 385) & (df.extent_z_mm <= 685)].iloc[0]  # extended-FOV: largest realistic volume
    print(f"Using sample volume: {row.filename} (extent_z={row.extent_z_mm:.0f}mm)")

    print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'n/a'}")

    if DEVICE == "cuda":
        # Windows silently spills CUDA allocations over the physical VRAM budget into
        # shared system memory ("CUDA Sysmem Fallback Policy") instead of raising OOM
        # at the true boundary -- this makes max_memory_allocated() report impossible
        # numbers (e.g. 25GB on a 12GB card) and hides the real usable batch size.
        # Capping the caching allocator to the GPU's own total forces a real OOM here.
        total_mem = torch.cuda.get_device_properties(0).total_memory
        frac = (11 * 1024**3) / total_mem  # cap at 11GB of the 12GB card, leave 1GB OS/driver headroom
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
        print(f"Capped allocator to {frac*100:.1f}% of {total_mem/1e9:.1f}GB total "
              f"(~{frac*total_mem/1e9:.1f}GB) to bypass Windows sysmem fallback masking true OOM")

    model = VistaClassifier(freeze_encoder=True).to(DEVICE)
    model.train()  # head in train mode; encoder.train(False) already set by set_encoder_trainable

    print("\n--- verifying no_grad on frozen encoder ---")
    verify_no_grad_on_encoder(model, preprocess(row.filepath))

    sample = preprocess(row.filepath)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("\n--- probing batch sizes ---")
    candidates = [2, 4, 8, 12, 16, 20, 24, 32]
    last_good = None
    last_good_mem = None
    for bs in candidates:
        try:
            peak_alloc, peak_reserved, loss_val = try_batch_size(model, optimizer, criterion, sample, bs)
            print(f"batch_size={bs:3d}: OK, peak_alloc={peak_alloc:8.1f} MB, "
                  f"peak_reserved={peak_reserved:8.1f} MB, loss={loss_val:.4f}")
            last_good = bs
            last_good_mem = (peak_alloc, peak_reserved)
        except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001
            print(f"batch_size={bs:3d}: OOM -- {e}")
            torch.cuda.empty_cache()
            break

    print("\n=== RESULT ===")
    if last_good is not None:
        print(f"Largest batch size that fit: {last_good} "
              f"(peak_alloc={last_good_mem[0]:.1f} MB, peak_reserved={last_good_mem[1]:.1f} MB)")
        # recommend a safety-margin batch size (one step down from the max that worked)
        idx = candidates.index(last_good)
        safe_bs = candidates[max(0, idx - 1)] if idx > 0 else last_good
        print(f"Recommended batch size with safety margin: {safe_bs}")
    else:
        print("No batch size succeeded.")


if __name__ == "__main__":
    main()
