"""
Diagnose whether the body-mask fix (largest-tissue-component + fill_holes,
no closing) is over-correcting -- i.e. whether it's throwing away real
intra-abdominal gas along with the background halo it was meant to remove.

Hypothesis under test: binary_fill_holes only fills a background region if
it does NOT touch the volume's outer border. Free intraperitoneal air often
sits right against the abdominal wall (classically just under it, e.g.
subphrenic air) -- if the abdominal wall is thin or blurred by resampling
partial-volume effects, the "hole" containing that air can leak into a thin
low-HU path connecting to the background OUTSIDE the body, at which point
it's no longer a "hole" (it's part of the border-touching background
component) and fill_holes will NOT recover it. This would silently delete
real disease-relevant gas along with the background.

Fix under test: apply a binary closing (dilate then erode) to the
body-tissue threshold BEFORE labeling/fill_holes, to bridge exactly this
kind of thin leak path, then compare recovered voxel counts.
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
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (  # noqa: E402
    AUDIT_CSV, BODY_MASK_HU_THRESHOLD, EXTENDED_FOV_THRESHOLD_MM, ROI_BBOX_CSV,
    _crop_to_bbox_mm, _load_reoriented_resampled,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "scripts", "air_mask_qc")
AIR_THRESHOLD = -600.0
CLOSING_ITERATIONS = 3  # ~3 voxels = ~4.5mm at the 1.5mm isotropic resample spacing


def load_raw_hu(filepath: str, audit_df: pd.DataFrame, roi_df: pd.DataFrame) -> np.ndarray:
    """Reproduce dataset.py's pipeline up to (but not including) the final
    resize-to-patch_size, so this diagnostic runs at native cropped
    resolution -- the SAME resolution the real air-mask computation runs at."""
    row = audit_df[audit_df.filepath == filepath].iloc[0]
    data = _load_reoriented_resampled(filepath)

    if row["extent_z_mm"] >= EXTENDED_FOV_THRESHOLD_MM:
        roi_row = roi_df[roi_df.filepath == filepath].iloc[0]
        bbox_min_mm = (roi_row["bbox_min_x"], roi_row["bbox_min_y"], roi_row["bbox_min_z"])
        bbox_max_mm = (roi_row["bbox_max_x"], roi_row["bbox_max_y"], roi_row["bbox_max_z"])
        data = _crop_to_bbox_mm(data, bbox_min_mm, bbox_max_mm)

    raw_hu = torch.as_tensor(np.asarray(data), dtype=torch.float32).squeeze(0)  # [X, Y, Z]
    return raw_hu.numpy()


def body_mask_naive(arr: np.ndarray, threshold: float = BODY_MASK_HU_THRESHOLD) -> np.ndarray:
    """The CURRENTLY-COMMITTED approach: threshold, largest component, fill holes. NO closing."""
    body_binary = arr > threshold
    labeled, num = ndimage.label(body_binary)
    if num == 0:
        return np.ones_like(arr, dtype=bool)
    sizes = ndimage.sum(body_binary, labeled, index=range(1, num + 1))
    largest = int(np.argmax(sizes)) + 1
    body = labeled == largest
    return ndimage.binary_fill_holes(body)


def body_mask_closed(arr: np.ndarray, threshold: float = BODY_MASK_HU_THRESHOLD,
                      iterations: int = CLOSING_ITERATIONS) -> np.ndarray:
    """Candidate fix: binary closing BEFORE labeling, to bridge thin leak paths
    (e.g. a blurred/thin abdominal wall right next to free air) that would
    otherwise connect an internal gas pocket to the background exterior."""
    body_binary = arr > threshold
    struct = ndimage.generate_binary_structure(3, 1)
    closed = ndimage.binary_closing(body_binary, structure=struct, iterations=iterations)
    labeled, num = ndimage.label(closed)
    if num == 0:
        return np.ones_like(arr, dtype=bool)
    sizes = ndimage.sum(closed, labeled, index=range(1, num + 1))
    largest = int(np.argmax(sizes)) + 1
    body = labeled == largest
    return ndimage.binary_fill_holes(body)


def analyze(filepath: str, label: str, audit_df, roi_df):
    arr = load_raw_hu(filepath, audit_df, roi_df)
    pre_fix_mask = arr < AIR_THRESHOLD  # the ORIGINAL, unrestricted threshold

    naive_body = body_mask_naive(arr)
    closed_body = body_mask_closed(arr)

    naive_air = pre_fix_mask & naive_body
    closed_air = pre_fix_mask & closed_body

    n_pre = pre_fix_mask.sum()
    n_naive = naive_air.sum()
    n_closed = closed_air.sum()
    n_outside_naive_body = pre_fix_mask.sum() - (pre_fix_mask & naive_body).sum()

    print(f"\n=== {label}: {os.path.basename(filepath)} ===")
    print(f"  volume shape: {arr.shape}")
    print(f"  pre-fix mask (raw <{AIR_THRESHOLD:.0f} HU, unrestricted): {n_pre} voxels "
          f"({100*n_pre/arr.size:.2f}% of volume)")
    print(f"  naive body-restricted (no closing):  {n_naive} voxels "
          f"({100*n_naive/n_pre:.2f}% of pre-fix mask kept, {100*(1-n_naive/n_pre):.2f}% removed)")
    print(f"  closed body-restricted (candidate fix): {n_closed} voxels "
          f"({100*n_closed/n_pre:.2f}% of pre-fix mask kept, {100*(1-n_closed/n_pre):.2f}% removed)")
    recovered = n_closed - n_naive
    print(f"  RECOVERED by closing: {recovered} voxels ({100*recovered/n_pre:.2f}% of the original mask) "
          f"-- these were real gas voxels the naive fix was incorrectly discarding as background"
          if recovered > 0 else "  closing recovered nothing (naive fix may have been fine here)")

    return arr, pre_fix_mask, naive_body, closed_body, naive_air, closed_air


def render_triple(arr, pre_fix_mask, naive_body, closed_air, title, out_path):
    c = [s // 2 for s in arr.shape]
    intensity = np.clip((arr + 963.82) / (1053.68 + 963.82), 0, 1)

    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    views = [
        (lambda v: v[c[0], :, :].T, "sagittal"),
        (lambda v: v[:, c[1], :].T, "coronal"),
        (lambda v: v[:, :, c[2]].T, "axial"),
    ]
    rows = [
        (pre_fix_mask, "PRE-FIX air mask (raw <-600 HU, unrestricted)"),
        (naive_body, "NAIVE body mask (threshold+largest-CC+fill, no closing)"),
        (closed_air, "POST-FIX air mask (closing-based body mask intersected)"),
    ]
    for row_idx, (overlay, row_title) in enumerate(rows):
        for col_idx, (slicer, name) in enumerate(views):
            ax = axes[row_idx, col_idx]
            ax.imshow(slicer(intensity), cmap="gray", origin="lower", vmin=0, vmax=1)
            ov = slicer(overlay.astype(float))
            ax.imshow(ov, cmap="autumn", origin="lower", alpha=0.5 * (ov > 0.1), vmin=0, vmax=1)
            ax.set_title(f"{row_title}\n{name}", fontsize=9)
            ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    audit_df = pd.read_csv(AUDIT_CSV)
    roi_df = pd.read_csv(ROI_BBOX_CSV)

    pos_fp = audit_df[audit_df.label == "positive"].sample(n=1, random_state=3).iloc[0]["filepath"]
    neg_fp = audit_df[audit_df.label == "negative"].sample(n=1, random_state=3).iloc[0]["filepath"]

    os.makedirs(OUT_DIR, exist_ok=True)
    for fp, label in [(pos_fp, "positive"), (neg_fp, "negative")]:
        arr, pre_fix_mask, naive_body, closed_body, naive_air, closed_air = analyze(fp, label, audit_df, roi_df)
        filename = os.path.basename(fp)
        out_path = os.path.join(OUT_DIR, f"diagnose__{label}__{filename.replace('.nii.gz', '')}.png")
        render_triple(arr, pre_fix_mask, naive_body, closed_air,
                      f"{filename} ({label}): pre-fix vs naive-body-mask vs closing-fixed", out_path)
        print(f"  saved diagnostic panel: {out_path}")


if __name__ == "__main__":
    main()
