"""
End-to-end validation of the fixed-FOV pipeline (adaptive ROI crop -> fixed
3mm/voxel spacing -> pad/crop to 224^3, NO resize) against all 23 real GT
free-air segmentations: does the FULL pipeline -- as PneumoDataset will
actually run it -- ever lose real GT signal, at the actual configured
spacing/patch_size (not just the raw bbox coverage check already done in
validate_adaptive_roi_crop.py).

Runs the GT mask through the identical crop -> resample -> pad/crop chain
(via monai's Spacing, matching dataset.py's own transforms exactly) so
resample-interpolation effects on a binary mask are captured too, not just
integer voxel-bbox arithmetic.
"""
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import MetaTensor
from monai.transforms import Orientation, Spacing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_roi_crop import bbox_vox_to_mm, compute_tight_3axis_bbox, load_ras_native  # noqa: E402
from dataset import FIXED_FOV_PATCH_SIZE, FIXED_FOV_SPACING, _crop_to_bbox_mm, _pad_or_crop_to_size  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(ROOT, "merged", "GT")
CT_DIR = os.path.join(ROOT, "merged", "Pneumo_Positive")
OUT_CSV = os.path.join(ROOT, "scripts", "fixed_fov_pipeline_vs_gt.csv")


def load_gt_as_metatensor(gt_path, ct_affine):
    gt_img = nib.load(gt_path)
    gt_data = gt_img.get_fdata(dtype=np.float32)[None]
    affine = torch.as_tensor(gt_img.affine, dtype=torch.float64)
    mt = MetaTensor(torch.as_tensor(gt_data, dtype=torch.float32), affine=affine)
    mt = Orientation(axcodes="RAS")(mt)
    mt = Spacing(pixdim=FIXED_FOV_SPACING, mode="nearest")(mt)  # nearest: keep binary, no blending
    return mt


def main():
    gt_files = sorted(os.listdir(GT_DIR))
    rows = []

    for fname in gt_files:
        ct_path = os.path.join(CT_DIR, fname)
        gt_path = os.path.join(GT_DIR, fname)
        if not os.path.exists(ct_path):
            continue

        # 1. adaptive crop on native CT -> bbox in mm (same as production path)
        hu_array, native_affine, native_zooms = load_ras_native(ct_path)
        bbox, _, _, _ = compute_tight_3axis_bbox(hu_array, native_zooms)
        bbox_min_mm, bbox_max_mm = bbox_vox_to_mm(native_affine, bbox)

        # 2. resample GT to FIXED_FOV_SPACING, crop to the same mm bbox, pad/crop to patch size
        gt_mt = load_gt_as_metatensor(gt_path, native_affine)
        gt_cropped = _crop_to_bbox_mm(gt_mt, bbox_min_mm, bbox_max_mm)
        gt_tensor = torch.as_tensor(np.asarray(gt_cropped), dtype=torch.float32)  # [1, X, Y, Z]
        n_gt_before_pad = float((gt_tensor > 0.5).sum())

        gt_final = _pad_or_crop_to_size(gt_tensor, FIXED_FOV_PATCH_SIZE, pad_value=0.0, bias_high_axis=2)
        n_gt_after_pad = float((gt_final > 0.5).sum())

        # 3. original GT voxel count (native resolution, un-resampled) for the frac_lost denominator
        gt_native = nib.as_closest_canonical(nib.load(gt_path)).get_fdata().astype(bool)
        n_gt_native = int(gt_native.sum())
        if n_gt_native == 0:
            continue

        # resample changes voxel COUNT (different grid spacing) even with zero loss, so compare
        # "voxels present after resample+crop" vs "voxels present after resample, before crop" --
        # isolates what the CROP+PAD step itself does, not resample-interpolation rounding
        n_gt_after_resample_only = float((torch.as_tensor(np.asarray(gt_mt), dtype=torch.float32) > 0.5).sum())
        frac_lost_to_crop = 1.0 - (n_gt_before_pad / n_gt_after_resample_only) if n_gt_after_resample_only > 0 else 0.0
        frac_lost_to_pad_crop = 1.0 - (n_gt_after_pad / n_gt_before_pad) if n_gt_before_pad > 0 else 0.0

        rows.append({
            "filename": fname, "n_gt_native": n_gt_native,
            "n_gt_after_resample": n_gt_after_resample_only,
            "n_gt_after_roi_crop": n_gt_before_pad, "n_gt_after_pad_crop": n_gt_after_pad,
            "frac_lost_to_roi_crop": frac_lost_to_crop, "frac_lost_to_pad_crop": frac_lost_to_pad_crop,
        })
        flag = ""
        if frac_lost_to_crop > 0.001:
            flag += "  <-- ROI CROP LOSES SIGNAL"
        if frac_lost_to_pad_crop > 0.001:
            flag += "  <-- PAD/CROP-TO-224 LOSES SIGNAL"
        print(f"{fname}: roi_crop_loss={frac_lost_to_crop:.2%}, pad_crop_loss={frac_lost_to_pad_crop:.2%}{flag}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n=== SUMMARY (n={len(df)}) ===")
    print(f"Cases losing signal to ROI crop step: {(df['frac_lost_to_roi_crop'] > 0.001).sum()} / {len(df)}")
    print(f"Cases losing signal to pad/crop-to-224 step: {(df['frac_lost_to_pad_crop'] > 0.001).sum()} / {len(df)}")
    print(f"Max frac_lost_to_roi_crop: {df['frac_lost_to_roi_crop'].max():.2%}")
    print(f"Max frac_lost_to_pad_crop: {df['frac_lost_to_pad_crop'].max():.2%}")
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
