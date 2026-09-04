"""
Visual QC for train-only 3D augmentation, same style as the earlier ROI-crop
QC panels. Specifically checking the two failure modes flagged before trusting
any augmented training run:

  1. A rotation pushing already-cropped anatomy (fixed 280mm z-window,
     extended-FOV volumes only) outside the frame -- would show as content
     missing at an edge that's present in the unaugmented version.
  2. Border-padding artifacts from grid_sample's rotation that could look like
     a sharp anatomical edge -- would show as a visible straight/curved seam
     not present in the unaugmented original.

Renders original vs. 4 independently-sampled augmented copies, axial/coronal/
sagittal through the volume center, for a handful of extended-FOV (crop-
affected) volumes -- these are the ones with the least margin, so any bug
shows up here first.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment import augment_volume  # noqa: E402
from dataset import EXTENDED_FOV_THRESHOLD_MM, PneumoDataset  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
OUT_DIR = os.path.join(ROOT, "scripts", "aug_qc")
N_VOLUMES = 3
N_AUG_COPIES = 4


def render_panel(original, augmented_list, title, out_path):
    n_rows = 1 + len(augmented_list)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))

    def plot_row(row_idx, vol, label):
        c = [s // 2 for s in vol.shape]
        views = [
            (vol[c[0], :, :].T, "sagittal"),
            (vol[:, c[1], :].T, "coronal"),
            (vol[:, :, c[2]].T, "axial"),
        ]
        for col_idx, (sl, name) in enumerate(views):
            ax = axes[row_idx, col_idx]
            ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.set_title(f"{label} {name}")
            ax.axis("off")

    plot_row(0, original, "ORIGINAL")
    for i, aug in enumerate(augmented_list):
        plot_row(1 + i, aug, f"AUG#{i+1}")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    audit = pd.read_csv(AUDIT_CSV)
    extended = audit[audit["extent_z_mm"] >= EXTENDED_FOV_THRESHOLD_MM]
    picks = extended.sample(n=min(N_VOLUMES, len(extended)), random_state=7)["filepath"].tolist()

    ds = PneumoDataset(picks, patch_size=(160, 160, 160), augment=False)

    for i, filepath in enumerate(picks):
        x, y, fp = ds[i]
        original = x[0].numpy()

        augmented_list = []
        for _ in range(N_AUG_COPIES):
            x_aug = augment_volume(x.clone())
            augmented_list.append(x_aug[0].numpy())

        filename = os.path.basename(fp)
        title = f"{filename} (extended-FOV, fixed 280mm z-crop applied) label={'pos' if y==1 else 'neg'}"
        out_path = os.path.join(OUT_DIR, f"aug_qc__{filename.replace('.nii.gz', '')}.png")
        render_panel(original, augmented_list, title, out_path)
        print(f"[{i+1}/{len(picks)}] {filename}: saved {out_path}")

    print(f"\nAll QC panels saved to {OUT_DIR}. Inspect for:")
    print("  - content missing at an edge in AUG that's present in ORIGINAL (rotation pushed anatomy out)")
    print("  - a visible straight/curved seam not present in ORIGINAL (border-padding artifact)")


if __name__ == "__main__":
    main()
