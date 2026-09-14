"""
Visualize the adaptive ROI crop bounding box overlaid on the ORIGINAL native
CT, all 3 views -- shows exactly what region gets kept before any resample/
pad happens, so the crop itself can be sanity-checked directly.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
ADAPTIVE_ROI_CSV = os.path.join(ROOT, "scripts", "roi_bboxes_adaptive.csv")
OUT_DIR = os.path.join(ROOT, "scripts", "air_mask_qc")
HU_A_MIN, HU_A_MAX = -963.8247715525971, 1053.678477684517


def render_case(fp, label, roi_row):
    img = nib.as_closest_canonical(nib.load(fp))
    affine = img.affine
    hu = img.get_fdata(dtype=np.float32)
    shape = hu.shape
    scaled = np.clip(hu, HU_A_MIN, HU_A_MAX)
    scaled = (scaled - HU_A_MIN) / (HU_A_MAX - HU_A_MIN)

    bbox_min_mm = (roi_row["bbox_min_x"], roi_row["bbox_min_y"], roi_row["bbox_min_z"])
    bbox_max_mm = (roi_row["bbox_max_x"], roi_row["bbox_max_y"], roi_row["bbox_max_z"])
    inv = np.linalg.inv(affine)
    corners_mm = np.array([
        [bbox_min_mm[0], bbox_min_mm[1], bbox_min_mm[2], 1],
        [bbox_max_mm[0], bbox_max_mm[1], bbox_max_mm[2], 1],
    ])
    corners_vox = (inv @ corners_mm.T).T[:, :3]
    vox_min = np.clip(np.floor(corners_vox.min(axis=0)).astype(int), 0, np.array(shape) - 1)
    vox_max = np.clip(np.ceil(corners_vox.max(axis=0)).astype(int), 0, np.array(shape) - 1)

    center = ((vox_min + vox_max) // 2).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # sagittal (fix x at bbox center), show y/z box
    ax = axes[0]
    ax.imshow(scaled[center[0], :, :].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    rect = patches.Rectangle((vox_min[1], vox_min[2]), vox_max[1] - vox_min[1], vox_max[2] - vox_min[2],
                              linewidth=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"sagittal x={center[0]}")
    ax.axis("off")

    # coronal (fix y), show x/z box
    ax = axes[1]
    ax.imshow(scaled[:, center[1], :].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    rect = patches.Rectangle((vox_min[0], vox_min[2]), vox_max[0] - vox_min[0], vox_max[2] - vox_min[2],
                              linewidth=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"coronal y={center[1]}")
    ax.axis("off")

    # axial (fix z), show x/y box
    ax = axes[2]
    ax.imshow(scaled[:, :, center[2]].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    rect = patches.Rectangle((vox_min[0], vox_min[1]), vox_max[0] - vox_min[0], vox_max[1] - vox_min[1],
                              linewidth=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"axial z={center[2]}")
    ax.axis("off")

    x_mm = bbox_max_mm[0] - bbox_min_mm[0]
    y_mm = bbox_max_mm[1] - bbox_min_mm[1]
    z_mm = bbox_max_mm[2] - bbox_min_mm[2]
    filename = os.path.basename(fp)
    fig.suptitle(f"{filename} label={label} | native shape={shape} | "
                 f"crop extent (mm): x={x_mm:.0f}, y={y_mm:.0f}, z={z_mm:.0f}")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"roi_bbox_overlay__{label}__{filename.replace('.nii.gz', '')}.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[{label}] {filename}: crop extent x={x_mm:.0f}mm y={y_mm:.0f}mm z={z_mm:.0f}mm -> {out_path}")
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    audit = pd.read_csv(AUDIT_CSV)
    roi = pd.read_csv(ADAPTIVE_ROI_CSV)

    pos_fp = audit[audit.label == "positive"].sample(n=1, random_state=3).iloc[0]["filepath"]
    neg_fp = audit[audit.label == "negative"].sample(n=1, random_state=3).iloc[0]["filepath"]

    for fp, label in [(pos_fp, "positive"), (neg_fp, "negative")]:
        row = roi[roi.filepath == fp]
        if row.empty:
            print(f"SKIP {fp}: not in {ADAPTIVE_ROI_CSV} (likely abdomen-only, not cropped)")
            continue
        render_case(fp, label, row.iloc[0])


if __name__ == "__main__":
    main()
