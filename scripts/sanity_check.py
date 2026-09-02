"""
Sanity check: run the planned preprocessing (reorient -> resample 1.5mm iso ->
VISTA3D intensity window -> crude resize/pad to 128^3) on 3 real volumes
through VistaClassifier, before building the full dataloader/ROI-crop
pipeline. Checks: no errors, no NaN/Inf, actual peak GPU memory.
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

    # add channel dim: [1, X, Y, Z], wrap as MetaTensor so affine travels with the data
    data = data[None]
    data = MetaTensor(torch.as_tensor(data, dtype=torch.float32), affine=torch.as_tensor(affine, dtype=torch.float64))

    orient = Orientation(axcodes="RAS")
    data = orient(data)

    spacing = Spacing(pixdim=RESAMPLE_SPACING, mode="bilinear")
    data = spacing(data)

    data = torch.as_tensor(np.asarray(data), dtype=torch.float32)

    # VISTA3D exact intensity window -> [0,1]
    data = torch.clamp(data, HU_A_MIN, HU_A_MAX)
    data = (data - HU_A_MIN) / (HU_A_MAX - HU_A_MIN)

    # crude resize/pad to 128^3 (placeholder for real ROI-crop logic)
    data = data.unsqueeze(0)  # [1, 1, X, Y, Z]
    data = F.interpolate(data, size=PATCH_SIZE, mode="trilinear", align_corners=False)

    return data  # [1, 1, 128, 128, 128]


def run_case(name: str, filepath: str, model: VistaClassifier):
    print(f"\n=== {name}: {os.path.basename(filepath)} ===")
    try:
        x = preprocess(filepath)
        print(f"preprocessed shape: {tuple(x.shape)}, "
              f"min={x.min().item():.4f}, max={x.max().item():.4f}, "
              f"mean={x.mean().item():.4f}")
        assert not torch.isnan(x).any(), "NaN in preprocessed input"
        assert not torch.isinf(x).any(), "Inf in preprocessed input"

        x = x.to(DEVICE)
        model = model.to(DEVICE)

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        with torch.no_grad():
            stages = model.encoder(x)
            deepest = stages[-1]
            pooled = model.pool(deepest)
            logit = model(x)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            peak_reserved_mb = torch.cuda.max_memory_reserved() / 1e6
        else:
            peak_mb = peak_reserved_mb = float("nan")

        pooled_nan = torch.isnan(pooled).any().item()
        pooled_inf = torch.isinf(pooled).any().item()
        logit_nan = torch.isnan(logit).any().item()
        logit_inf = torch.isinf(logit).any().item()

        print(f"deepest stage shape: {tuple(deepest.shape)}")
        print(f"pooled features shape: {tuple(pooled.shape)}, "
              f"NaN={pooled_nan}, Inf={pooled_inf}, "
              f"min={pooled.min().item():.4f}, max={pooled.max().item():.4f}")
        print(f"logit: {logit.item():.6f}, NaN={logit_nan}, Inf={logit_inf}")
        print(f"peak GPU memory allocated: {peak_mb:.1f} MB, reserved: {peak_reserved_mb:.1f} MB")

        ok = not (pooled_nan or pooled_inf or logit_nan or logit_inf)
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
        return ok, peak_mb
    except Exception as e:  # noqa: BLE001
        print(f"RESULT: ERROR -- {type(e).__name__}: {e}")
        raise


def main():
    df = pd.read_csv(AUDIT_CSV)
    abd = df[(df.extent_z_mm >= 150) & (df.extent_z_mm < 375)].iloc[0]
    ext = df[df.extent_z_mm >= 375].iloc[0]
    thick = df[df.spacing_z > 10].iloc[0]

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = VistaClassifier(freeze_encoder=True)
    model.eval()

    results = {}
    for name, row in [("abdomen-only", abd), ("extended-FOV", ext), ("thick-slice", thick)]:
        ok, peak_mb = run_case(
            f"{name} (extent_z={row.extent_z_mm:.0f}mm, spacing_z={row.spacing_z:.1f}mm)",
            row.filepath,
            model,
        )
        results[name] = (ok, peak_mb)

    print("\n=== SUMMARY ===")
    for name, (ok, peak_mb) in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}, peak GPU mem: {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
