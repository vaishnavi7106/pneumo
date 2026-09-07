"""
Evaluate the air-mask channel (raw HU < threshold, restricted to the body
mask) against ALL real GT free-air segmentations in merged/GT/ -- 23 cases,
never previously used in this project. Computed on each volume's NATIVE grid
(no resample/crop) so voxel grids match GT exactly, no interpolation needed.

Reports per-case and aggregate IoU/recall/precision, so a body-mask fix can
be judged against real ground truth instead of visual plausibility alone.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import BODY_MASK_HU_THRESHOLD  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(ROOT, "merged", "GT")
CT_DIR = os.path.join(ROOT, "merged", "Pneumo_Positive")
AIR_THRESHOLD = -600.0


def compute_body_mask(raw_hu_3d: torch.Tensor, body_threshold: float, closing_iters: int,
                       erode_iters: int) -> torch.Tensor:
    arr = raw_hu_3d.numpy()
    body_binary = arr > body_threshold
    struct = ndimage.generate_binary_structure(3, 1)
    if closing_iters > 0:
        body_binary = ndimage.binary_closing(body_binary, structure=struct, iterations=closing_iters)

    labeled, num = ndimage.label(body_binary)
    if num == 0:
        return torch.ones_like(raw_hu_3d, dtype=torch.float32)
    sizes = ndimage.sum(body_binary, labeled, index=range(1, num + 1))
    largest = int(np.argmax(sizes)) + 1
    body_mask = labeled == largest
    body_mask = ndimage.binary_fill_holes(body_mask)
    if erode_iters > 0:
        body_mask = ndimage.binary_erosion(body_mask, structure=struct, iterations=erode_iters)
    return torch.as_tensor(body_mask, dtype=torch.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--closing-iters", type=int, default=3)
    p.add_argument("--erode-iters", type=int, default=1)
    args = p.parse_args()

    gt_files = sorted(os.listdir(GT_DIR))
    print(f"Evaluating {len(gt_files)} GT cases -- closing_iters={args.closing_iters}, "
          f"erode_iters={args.erode_iters}\n")

    rows = []
    for fname in gt_files:
        ct_path = os.path.join(CT_DIR, fname)
        gt_path = os.path.join(GT_DIR, fname)
        if not os.path.exists(ct_path):
            print(f"  SKIP {fname}: no matching CT in {CT_DIR}")
            continue

        raw_hu = nib.load(ct_path).get_fdata(dtype=np.float32)
        gt = nib.load(gt_path).get_fdata().astype(bool)

        body_mask = compute_body_mask(torch.as_tensor(raw_hu), BODY_MASK_HU_THRESHOLD,
                                       args.closing_iters, args.erode_iters).numpy().astype(bool)
        air_mask = (raw_hu < AIR_THRESHOLD) & body_mask

        intersection = (air_mask & gt).sum()
        union = (air_mask | gt).sum()
        iou = intersection / union if union > 0 else float("nan")
        recall = intersection / gt.sum() if gt.sum() > 0 else float("nan")
        precision = intersection / air_mask.sum() if air_mask.sum() > 0 else float("nan")

        rows.append({"filename": fname, "gt_voxels": int(gt.sum()), "air_mask_voxels": int(air_mask.sum()),
                     "iou": iou, "recall": recall, "precision": precision})
        print(f"  {fname}: IoU={iou:.4f}  recall={recall:.4f}  precision={precision:.4f}  "
              f"(gt={gt.sum()}, air_mask={air_mask.sum()})")

    df = pd.DataFrame(rows)
    print(f"\n=== AGGREGATE (n={len(df)}) ===")
    for metric in ["iou", "recall", "precision"]:
        print(f"  {metric}: mean={df[metric].mean():.4f}  median={df[metric].median():.4f}  "
              f"std={df[metric].std():.4f}")

    out_csv = os.path.join(ROOT, "scripts", "air_mask_qc_extended",
                            f"gt_eval_closing{args.closing_iters}_erode{args.erode_iters}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-case results to {out_csv}")


if __name__ == "__main__":
    main()
