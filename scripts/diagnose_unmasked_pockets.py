"""
Diagnostic: is the "some air pockets aren't masked" pattern seen in
qc_air_mask.py a REAL recall gap (HU threshold or body-mask exclusion
missing the pocket entirely), or a DISPLAY artifact (the pocket is masked,
but trilinear resize to the small QC patch size + the alpha>0.1 display
cutoff makes it invisible)?

Replicates dataset.py's air-mask computation up to the point BEFORE the
resize-to-patch-size step (i.e. at the resampled/cropped, native mask
resolution), and renders it with a hard 0/1 overlay (no display threshold
ambiguity possible at this stage). Compares side-by-side against the
post-resize soft mask actually fed to the model, at the same anatomical
slice.

If a pocket is missing even in the PRE-resize panel: real recall gap
(thresholding/body-mask problem). If it's present pre-resize but vanishes
post-resize: display/downsampling artifact.
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
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (  # noqa: E402
    AUDIT_CSV, EXTENDED_FOV_THRESHOLD_MM, HU_A_MAX, HU_A_MIN, ROI_BBOX_CSV,
    _compute_body_mask, _crop_to_bbox_mm, _load_reoriented_resampled,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "scripts", "air_mask_qc")
AIR_MASK_THRESHOLD = -600.0
PATCH_SIZE = (160, 160, 160)  # matches qc_air_mask.py's patch size


def compute_pre_resize_mask(filepath: str):
    audit = pd.read_csv(AUDIT_CSV)
    row = audit.set_index("filepath").loc[filepath]

    roi = {}
    if os.path.exists(ROI_BBOX_CSV):
        roi_df = pd.read_csv(ROI_BBOX_CSV)
        for _, r in roi_df.iterrows():
            if r.get("status") == "ok":
                roi[r["filepath"]] = (
                    (r["bbox_min_x"], r["bbox_min_y"], r["bbox_min_z"]),
                    (r["bbox_max_x"], r["bbox_max_y"], r["bbox_max_z"]),
                )

    data = _load_reoriented_resampled(filepath)
    if row["extent_z_mm"] >= EXTENDED_FOV_THRESHOLD_MM:
        bbox_min_mm, bbox_max_mm = roi[filepath]
        data = _crop_to_bbox_mm(data, bbox_min_mm, bbox_max_mm)

    raw_hu = torch.as_tensor(np.asarray(data), dtype=torch.float32)  # [1, X, Y, Z]

    x = torch.clamp(raw_hu, HU_A_MIN, HU_A_MAX)
    x = (x - HU_A_MIN) / (HU_A_MAX - HU_A_MIN)  # [1, X, Y, Z], intensity channel, native resampled res

    raw_hu_3d = raw_hu.squeeze(0)  # [X, Y, Z]
    body_mask_3d = _compute_body_mask(raw_hu_3d)
    air_binary_native = (raw_hu_3d < AIR_MASK_THRESHOLD) & (body_mask_3d > 0.5)  # [X, Y, Z], hard 0/1

    # what the model actually receives: intensity + soft mask, both resized to patch size
    intensity_resized = F.interpolate(x.unsqueeze(0), size=PATCH_SIZE, mode="trilinear",
                                       align_corners=False).squeeze(0).squeeze(0).numpy()
    mask_resized = F.interpolate(air_binary_native.float().unsqueeze(0).unsqueeze(0), size=PATCH_SIZE,
                                  mode="trilinear", align_corners=False).squeeze(0).squeeze(0).numpy()

    return {
        "intensity_native": x.squeeze(0).numpy(),          # native resampled resolution
        "air_binary_native": air_binary_native.numpy(),    # native resampled resolution, hard 0/1
        "intensity_resized": intensity_resized,             # what the model sees (post-resize)
        "mask_resized": mask_resized,                       # what the model sees (post-resize, soft)
    }


def render_comparison(result, title, out_path):
    intensity_native = result["intensity_native"]
    air_native = result["air_binary_native"]
    intensity_resized = result["intensity_resized"]
    mask_resized = result["mask_resized"]

    c_native = [s // 2 for s in intensity_native.shape]
    c_resized = [s // 2 for s in intensity_resized.shape]

    views = [
        ("sagittal", lambda a, c: a[c[0], :, :].T, 0),
        ("coronal", lambda a, c: a[:, c[1], :].T, 1),
        ("axial", lambda a, c: a[:, :, c[2]].T, 2),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    for col, (name, slicer, _axis) in enumerate(views):
        ct_native = slicer(intensity_native, c_native)
        mask_native_slice = slicer(air_native.astype(np.float32), c_native)
        axes[0, col].imshow(ct_native, cmap="gray", origin="lower", vmin=0, vmax=1)
        # hard mask, alpha=0.5 wherever mask==1 -- no display threshold ambiguity possible
        axes[0, col].imshow(mask_native_slice, cmap="autumn", origin="lower",
                             alpha=0.5 * (mask_native_slice > 0), vmin=0, vmax=1)
        axes[0, col].set_title(f"PRE-resize (native res) {name}")
        axes[0, col].axis("off")

        ct_resized = slicer(intensity_resized, c_resized)
        mask_resized_slice = slicer(mask_resized, c_resized)
        axes[1, col].imshow(ct_resized, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, col].imshow(mask_resized_slice, cmap="autumn", origin="lower",
                             alpha=0.5 * (mask_resized_slice > 0.1), vmin=0, vmax=1)
        axes[1, col].set_title(f"POST-resize (model input, {PATCH_SIZE[0]}^3) {name}")
        axes[1, col].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    cases = [
        ("20251028107-1.nii.gz", "negative"),  # the original case with the missing pockets
    ]
    audit = pd.read_csv(AUDIT_CSV)
    audit["basename"] = audit["filepath"].apply(os.path.basename)

    for basename, label in cases:
        row = audit[audit["basename"] == basename]
        if row.empty:
            print(f"WARNING: {basename} not found in audit CSV, skipping")
            continue
        filepath = row.iloc[0]["filepath"]

        result = compute_pre_resize_mask(filepath)
        native_frac = result["air_binary_native"].mean()
        resized_frac = (result["mask_resized"] > 0.5).mean()

        title = (f"{basename} label={label} | native air fraction={native_frac:.4f}, "
                 f"post-resize air fraction (>0.5)={resized_frac:.4f}")
        out_path = os.path.join(OUT_DIR, f"diagnose_unmasked__{label}__{basename.replace('.nii.gz', '')}.png")
        render_comparison(result, title, out_path)
        print(f"[{label}] {basename}: native_frac={native_frac:.4f}, resized_frac={resized_frac:.4f} "
              f"-> saved {out_path}")


if __name__ == "__main__":
    main()
