"""
Visual comparison of our air-mask channel (raw HU < threshold, restricted to
the body-mask) against a REAL ground-truth free-air segmentation, found at
merged/GT/<filename> -- same voxel grid/affine as the CT, never previously
used in this project.

Renders three rows through the GT mask's centroid slice: our air-mask alone,
the real GT alone, and a TP/FP/FN overlay (green=both agree, red=our mask
flags it but GT doesn't [false positive, e.g. bowel gas], blue=GT flags it
but our mask misses it [false negative]).
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import HU_A_MAX, HU_A_MIN, _compute_body_mask  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "scripts", "air_mask_qc_extended")
AIR_THRESHOLD = -600.0


def main(case_id="20251028006-1"):
    ct_path = os.path.join(ROOT, "merged", "Pneumo_Positive", f"{case_id}.nii.gz")
    gt_path = os.path.join(ROOT, "merged", "GT", f"{case_id}.nii.gz")

    ct_img = nib.load(ct_path)
    raw_hu = ct_img.get_fdata(dtype=np.float32)
    gt = nib.load(gt_path).get_fdata().astype(bool)

    body_mask = _compute_body_mask(torch.as_tensor(raw_hu)).numpy().astype(bool)
    air_mask = (raw_hu < AIR_THRESHOLD) & body_mask

    intersection = air_mask & gt
    false_pos = air_mask & ~gt
    false_neg = gt & ~air_mask

    iou = intersection.sum() / (air_mask | gt).sum()
    recall = intersection.sum() / gt.sum()
    precision = intersection.sum() / air_mask.sum()

    intensity = np.clip((raw_hu - HU_A_MIN) / (HU_A_MAX - HU_A_MIN), 0, 1)
    c = np.argwhere(gt).mean(axis=0).astype(int).tolist()

    views = [
        (lambda v: v[c[0], :, :].T, "sagittal"),
        (lambda v: v[:, c[1], :].T, "coronal"),
        (lambda v: v[:, :, c[2]].T, "axial"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    for col, (slicer, name) in enumerate(views):
        ct_slice = slicer(intensity)

        axes[0, col].imshow(ct_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
        ov = slicer(air_mask.astype(float))
        axes[0, col].imshow(ov, cmap="autumn", origin="lower", alpha=0.55 * (ov > 0.1), vmin=0, vmax=1)
        axes[0, col].set_title(f"OUR air_mask {name}")
        axes[0, col].axis("off")

        axes[1, col].imshow(ct_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
        ov = slicer(gt.astype(float))
        axes[1, col].imshow(ov, cmap="cool", origin="lower", alpha=0.55 * (ov > 0.1), vmin=0, vmax=1)
        axes[1, col].set_title(f"REAL GT {name}")
        axes[1, col].axis("off")

        axes[2, col].imshow(ct_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
        rgb = np.zeros((*ct_slice.shape, 4))
        tp = slicer(intersection.astype(float)) > 0.5
        fp = slicer(false_pos.astype(float)) > 0.5
        fn = slicer(false_neg.astype(float)) > 0.5
        rgb[tp] = [0, 1, 0, 0.6]   # green = agree
        rgb[fp] = [1, 0, 0, 0.6]   # red = our mask only (false positive, e.g. bowel gas)
        rgb[fn] = [0, 0.4, 1, 0.6]  # blue = GT only (false negative, we're missing it)
        axes[2, col].imshow(rgb, origin="lower")
        axes[2, col].set_title(f"TP(green)/FP(red)/FN(blue) {name}")
        axes[2, col].axis("off")

    fig.suptitle(f"{case_id}: our air_mask vs. real GT free-air segmentation\n"
                 f"IoU={iou:.4f}  recall={recall:.4f}  precision={precision:.4f}")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"vs_gt__{case_id}.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"IoU={iou:.4f}, recall={recall:.4f}, precision={precision:.4f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("case_id", nargs="?", default="20251028006-1")
    args = p.parse_args()
    main(args.case_id)
