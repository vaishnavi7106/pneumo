"""
Broader QC pass for the body-mask-restricted air-mask fix: checks 10-15
volumes for two things a visual spot-check can't easily catch at scale:

  1. Air fraction should be small and sane (a handful of percent, not the
     20-40%+ seen before the body-mask fix -- that was background leaking in).
  2. The mask should not touch the volume's outer border -- a real intra-body
     gas pocket is, by definition, inside the body silhouette, so any
     mask voxel sitting exactly on the edge of the array is a strong signal
     the body mask failed for that volume (e.g. the body touches the FOV
     edge, or segmentation picked the wrong connected component).
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PneumoDataset  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
N_VOLUMES = 14


def touches_border(mask_3d: np.ndarray) -> bool:
    return bool(
        mask_3d[0, :, :].any() or mask_3d[-1, :, :].any() or
        mask_3d[:, 0, :].any() or mask_3d[:, -1, :].any() or
        mask_3d[:, :, 0].any() or mask_3d[:, :, -1].any()
    )


def main():
    audit = pd.read_csv(AUDIT_CSV)
    picks = audit.sample(n=min(N_VOLUMES, len(audit)), random_state=11)["filepath"].tolist()

    ds = PneumoDataset(picks, patch_size=(160, 160, 160), air_mask_threshold=-600.0, use_cache=False)

    print(f"{'filename':40s} {'label':10s} {'air_frac':>10s} {'touches_border':>15s}")
    flagged = []
    for i, fp in enumerate(picks):
        x, y, _ = ds[i]
        mask = (x[1].numpy() > 0.5)
        frac = mask.mean()
        border = touches_border(mask)
        label = "positive" if y == 1 else "negative"
        filename = os.path.basename(fp)
        print(f"{filename:40s} {label:10s} {frac:10.4f} {str(border):>15s}")
        if border or frac > 0.10:
            flagged.append((filename, frac, border))

    print(f"\n{len(picks)} volumes checked.")
    if flagged:
        print(f"FLAGGED ({len(flagged)}) -- air_frac > 10% or mask touches border, inspect visually:")
        for filename, frac, border in flagged:
            print(f"  {filename}: air_frac={frac:.4f}, touches_border={border}")
    else:
        print("None flagged -- all volumes have a small, border-free air mask (body-mask fix holds generally).")


if __name__ == "__main__":
    main()
