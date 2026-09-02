"""
Directly check the SUPERIOR edge of the preprocessed crop (not just the
center slice) for evidence the diaphragm/liver dome survived cropping --
the concrete check the clinical concern actually calls for, since subphrenic
free air collects right at this boundary.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PneumoDataset  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
OUT_DIR = os.path.join(ROOT, "scripts", "diaphragm_check_png")

TARGET_FILES = [
    "20251028313-1.nii.gz", "20251028388-1.nii.gz", "20251028083-1.nii.gz",
    "20251028040-1.nii.gz", "20251028331-1.nii.gz", "20251028192-1.nii.gz",
    "20251028273-1.nii.gz", "20251028170-1.nii.gz",
]

os.makedirs(OUT_DIR, exist_ok=True)
audit = pd.read_csv(AUDIT_CSV)
rows = audit[audit["filename"].isin(TARGET_FILES)]

ds = PneumoDataset(rows["filepath"].tolist(), patch_size=(224, 224, 224))

for _, row in rows.iterrows():
    idx = ds.filepaths.index(row["filepath"])
    x, y, _ = ds[idx]
    vol = x.squeeze(0).numpy()  # [D(R), H(A), W(S)] -- S is array axis 2 after RAS orientation

    depth = vol.shape[2]
    # sample coronal slices near the TOP (superior/S-max) edge of the crop,
    # plus mid and bottom for context
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    sample_fracs = [0.98, 0.90, 0.75, 0.50]
    for ax, frac in zip(axes, sample_fracs):
        z = int(depth * frac) - 1
        z = max(0, min(z, depth - 1))
        # coronal-style view: fix a mid A-axis index, show (R, S) plane at various S depths
        # instead: show axial-style slice at this S index to look for lung/diaphragm air pockets
        sl = vol[:, :, z].T
        ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(f"S-index frac={frac:.2f} (z={z}/{depth-1})")
    fig.suptitle(f"{row['filename']} (label={row['label']}) -- axial slices near superior crop edge")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, row["filename"].replace(".nii.gz", "_topedge.png"))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    # also save a coronal slice showing the FULL superior-inferior extent in one view
    fig2, ax2 = plt.subplots(figsize=(6, 8))
    mid_a = vol.shape[1] // 2
    coronal = vol[:, mid_a, :].T
    ax2.imshow(coronal, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax2.axhline(depth - 1, color="red", linestyle="--", label="superior edge (crop boundary)")
    ax2.set_title(f"{row['filename']} coronal, full crop S-extent")
    ax2.legend()
    fig2.tight_layout()
    out_path2 = os.path.join(OUT_DIR, row["filename"].replace(".nii.gz", "_coronal_full.png"))
    fig2.savefig(out_path2, dpi=100)
    plt.close(fig2)

    print(f"{row['filename']}: saved top-edge + full-coronal views")

print(f"\nAll saved to {OUT_DIR}")
