"""
Memory probe at 224^3 patch size (vs. the 128^3 used elsewhere) -- same
VISTA3D encoder, same preprocessing pipeline, real training step (forward +
backward through the head + optimizer.step(), frozen encoder under
torch.no_grad()). Finds the largest practical batch size on the 3060's 12GB.
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
    return data.squeeze(0)  # [1, D, H, W]


def build_batch(sample: torch.Tensor, batch_size: int) -> torch.Tensor:
    return sample.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).contiguous()


def try_batch_size(model, optimizer, criterion, sample: torch.Tensor, batch_size: int):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = build_batch(sample, batch_size).to(DEVICE)
    y = torch.randint(0, 2, (batch_size,), dtype=torch.float32, device=DEVICE)

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    assert logits.requires_grad
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / 1e6
    peak_reserved = torch.cuda.max_memory_reserved() / 1e6
    return peak_alloc, peak_reserved, loss.item()


def main():
    df = pd.read_csv(AUDIT_CSV)
    row = df[(df.extent_z_mm >= 385) & (df.extent_z_mm <= 685)].iloc[0]
    print(f"Using sample volume: {row.filename} (extent_z={row.extent_z_mm:.0f}mm)")
    print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'n/a'}")

    if DEVICE == "cuda":
        # cap the allocator to the card's real budget -- Windows silently spills
        # overflow into shared system memory instead of raising OOM at the true
        # boundary, which corrupts naive memory probes (see earlier 128^3 probe)
        total_mem = torch.cuda.get_device_properties(0).total_memory
        frac = (11 * 1024**3) / total_mem
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
        print(f"Capped allocator to ~{frac*total_mem/1e9:.1f}GB of {total_mem/1e9:.1f}GB total")

    sample_128 = preprocess(row.filepath, (128, 128, 128))
    sample_224 = preprocess(row.filepath, PATCH_SIZE_224)
    print(f"\n224^3 sample shape: {tuple(sample_224.shape)}")

    model = VistaClassifier(freeze_encoder=True).to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("\n=== STEP 1: single 224^3, batch_size=1 ===")
    peak_alloc, peak_reserved, loss_val = try_batch_size(model, optimizer, criterion, sample_224, 1)
    print(f"batch_size=1: peak_alloc={peak_alloc:.1f} MB, peak_reserved={peak_reserved:.1f} MB, loss={loss_val:.4f}")

    print("\n=== STEP 2: sweeping batch size at 224^3 ===")
    candidates = [1, 2, 3, 4, 6, 8]
    last_good = None
    last_good_mem = None
    for bs in candidates:
        try:
            peak_alloc, peak_reserved, loss_val = try_batch_size(model, optimizer, criterion, sample_224, bs)
            print(f"batch_size={bs:3d}: OK, peak_alloc={peak_alloc:8.1f} MB, "
                  f"peak_reserved={peak_reserved:8.1f} MB, loss={loss_val:.4f}")
            last_good = bs
            last_good_mem = (peak_alloc, peak_reserved)
        except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001
            print(f"batch_size={bs:3d}: OOM -- {str(e).splitlines()[0]}")
            torch.cuda.empty_cache()
            break

    print("\n=== RESULT (224^3) ===")
    if last_good is not None:
        print(f"Largest batch size that fit: {last_good} "
              f"(peak_alloc={last_good_mem[0]:.1f} MB, peak_reserved={last_good_mem[1]:.1f} MB)")
        idx = candidates.index(last_good)
        safe_bs = candidates[max(0, idx - 1)] if idx > 0 else last_good
        print(f"Recommended batch size with safety margin: {safe_bs}")
    else:
        print("No batch size succeeded -- even batch_size=1 at 224^3 does not fit.")


if __name__ == "__main__":
    main()
