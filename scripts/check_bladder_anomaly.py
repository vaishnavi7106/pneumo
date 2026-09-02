"""
Visually inspect volumes flagged with a small bladder-label blob in their
top-5-slice band, to determine if it's genuine segmentation noise (a stray
mislabeled cluster near the esophagus/diaphragm) or a real orientation bug.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roi_localizer import segment_organ_bbox_mm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
OUT_DIR = os.path.join(ROOT, "scripts", "bladder_check_png")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ["20251028254-1.nii.gz", "20251028122-1.nii.gz", "20251028181-1.nii.gz"]

audit = pd.read_csv(AUDIT_CSV)

for fname in TARGETS:
    row = audit[audit.filename == fname].iloc[0]
    r = segment_organ_bbox_mm(row["filepath"], return_label_map=True)
    label_map = r["label_map"]
    shape = label_map.shape

    # find bladder (15) voxels overall, and specifically within the top-5 band
    bladder_mask = label_map == 15
    esoph_mask = label_map == 11
    top5 = label_map[:, :, shape[2]-5:shape[2]]
    top5_bladder = np.argwhere(top5 == 15)

    print(f"\n{fname}: total bladder voxels in whole volume={bladder_mask.sum()}, "
          f"total esophagus voxels={esoph_mask.sum()}, "
          f"bladder voxels in top-5 band={len(top5_bladder)}")

    if bladder_mask.sum() == 0:
        print("  (no bladder voxels anywhere -- shouldn't happen given json record)")
        continue

    all_bladder_coords = np.argwhere(bladder_mask)
    z_range = (all_bladder_coords[:, 2].min(), all_bladder_coords[:, 2].max())
    print(f"  bladder voxels span z-index range: {z_range} (volume z-depth={shape[2]})")
    print(f"  -> bladder appears at {'TOP' if z_range[1] > shape[2]*0.8 else 'BOTTOM' if z_range[0] < shape[2]*0.2 else 'MIDDLE'} "
          f"AND is it the ONLY location: {'yes, single tiny cluster' if z_range[1]-z_range[0] < 10 else 'spans a wide range -- suspicious'}")

    # plot: coronal slice through the bladder-labeled voxels, with bladder and esophagus highlighted
    mid_bladder_y = int(np.median(all_bladder_coords[:, 1]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    coronal = label_map[:, mid_bladder_y, :].T
    axes[0].imshow(coronal, cmap="gray", origin="lower", vmin=0, vmax=20)
    # overlay: red where bladder(15), blue where esophagus(11)
    overlay = np.zeros((*coronal.shape, 4))
    overlay[coronal == 15] = [1, 0, 0, 0.8]
    overlay[coronal == 11] = [0, 0.3, 1, 0.5]
    axes[0].imshow(overlay, origin="lower")
    axes[0].set_title(f"{fname}\ncoronal @ y={mid_bladder_y} (red=bladder id15, blue=esophagus id11)")
    axes[0].axhline(shape[2]-5, color="lime", linestyle="--", label="top-5-slice boundary")
    axes[0].legend()

    # zoomed view around the bladder voxels
    z0, z1 = max(0, z_range[0]-10), min(shape[2], z_range[1]+10)
    x0, x1 = max(0, all_bladder_coords[:,0].min()-15), min(shape[0], all_bladder_coords[:,0].max()+15)
    zoomed = label_map[x0:x1, mid_bladder_y, z0:z1].T
    axes[1].imshow(zoomed, cmap="tab20", origin="lower", vmin=0, vmax=20)
    axes[1].set_title(f"zoomed around bladder-labeled voxels (z={z0}-{z1}, x={x0}-{x1})")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, fname.replace(".nii.gz", "_bladder_check.png"))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  saved {out_path}")
