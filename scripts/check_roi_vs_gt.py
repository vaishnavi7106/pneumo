"""
Check whether the ROI crop (esp. the fixed 280mm z-window: 100mm inferior /
180mm superior from the core-organ centroid) is cutting off real free-air
signal -- using the 23 real GT segmentations in merged/GT/ as ground truth.

For each GT case that's extended-FOV (gets ROI-cropped): load the GT mask on
its own native grid, convert the cached ROI bbox (physical mm, from
roi_bboxes.csv) to voxel bounds on that same grid, and measure what fraction
of GT-positive (free-air) voxels fall OUTSIDE the bbox -- i.e. would be
silently cropped out during preprocessing, never even reaching the model.
Breaks the loss down by which side (inferior/superior in z, or x/y) the lost
voxels are on, since the z-window is asymmetric and the concern is
specifically about the lower abdomen / inferior margin being too tight.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import EXTENDED_FOV_THRESHOLD_MM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(ROOT, "merged", "GT")
CT_DIR = os.path.join(ROOT, "merged", "Pneumo_Positive")
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
ROI_CSV = os.path.join(ROOT, "scripts", "roi_bboxes.csv")
OUT_CSV = os.path.join(ROOT, "scripts", "roi_vs_gt_coverage.csv")


def mm_bbox_to_vox(affine, bbox_min_mm, bbox_max_mm):
    inv = np.linalg.inv(affine)
    corners_mm = np.array([
        [bbox_min_mm[0], bbox_min_mm[1], bbox_min_mm[2], 1],
        [bbox_max_mm[0], bbox_max_mm[1], bbox_max_mm[2], 1],
    ])
    corners_vox = (inv @ corners_mm.T).T[:, :3]
    vox_min = corners_vox.min(axis=0)
    vox_max = corners_vox.max(axis=0)
    return vox_min, vox_max


def main():
    audit = pd.read_csv(AUDIT_CSV)
    audit["basename"] = audit["filepath"].apply(os.path.basename)
    roi = pd.read_csv(ROI_CSV)
    roi["basename"] = roi["filepath"].apply(lambda p: os.path.basename(p))
    roi_ok = roi[roi["status"] == "ok"].set_index("basename")

    gt_files = sorted(os.listdir(GT_DIR))
    rows = []

    for fname in gt_files:
        ct_path = os.path.join(CT_DIR, fname)
        gt_path = os.path.join(GT_DIR, fname)
        if not os.path.exists(ct_path):
            print(f"SKIP {fname}: no matching CT")
            continue

        arow = audit[audit["basename"] == fname]
        if arow.empty:
            print(f"SKIP {fname}: not in audit CSV")
            continue
        extent_z = arow.iloc[0]["extent_z_mm"]

        gt_img = nib.load(gt_path)
        gt = gt_img.get_fdata().astype(bool)
        affine = gt_img.affine  # GT and CT share the same grid/affine (see eval_air_mask_vs_gt.py)

        n_gt = int(gt.sum())
        if n_gt == 0:
            print(f"SKIP {fname}: GT mask is empty")
            continue

        if extent_z < EXTENDED_FOV_THRESHOLD_MM:
            rows.append({"filename": fname, "extent_z_mm": extent_z, "cropped": False,
                         "n_gt_voxels": n_gt, "n_lost": 0, "frac_lost": 0.0,
                         "frac_lost_inferior": 0.0, "frac_lost_superior": 0.0,
                         "frac_lost_xy": 0.0})
            continue

        if fname not in roi_ok.index:
            print(f"WARNING {fname}: extended-FOV but no ok ROI bbox cached -- skipping")
            continue
        r = roi_ok.loc[fname]
        bbox_min_mm = (r["bbox_min_x"], r["bbox_min_y"], r["bbox_min_z"])
        bbox_max_mm = (r["bbox_max_x"], r["bbox_max_y"], r["bbox_max_z"])

        vox_min, vox_max = mm_bbox_to_vox(affine, bbox_min_mm, bbox_max_mm)
        # RAS-oriented mm axes map to this volume's own on-disk voxel axes via
        # the affine already -- but the sign of each voxel axis relative to
        # physical R/A/S direction depends on the affine, so min/max in voxel
        # space might be flipped per-axis; np.min/max above already handles that.

        coords = np.argwhere(gt)  # [N, 3] voxel indices (i, j, k) in this volume's own grid
        inside = np.all((coords >= np.floor(vox_min)) & (coords <= np.ceil(vox_max)), axis=1)
        n_lost = int((~inside).sum())
        frac_lost = n_lost / n_gt

        # figure out which physical axis corresponds to S (superior/inferior)
        # by finding which voxel axis has the largest |affine| component in
        # the mm z (S) row -- i.e. which voxel index moves S the fastest
        s_row = np.abs(affine[2, :3])  # mm-S sensitivity per voxel axis
        z_axis = int(np.argmax(s_row))
        z_sign = np.sign(affine[2, z_axis]) or 1.0

        lost_coords = coords[~inside]
        if n_lost > 0:
            z_vals = lost_coords[:, z_axis]
            # "superior" side = beyond vox_max on the S-increasing direction
            if z_sign > 0:
                lost_superior = (z_vals > vox_max[z_axis]).sum()
                lost_inferior = (z_vals < vox_min[z_axis]).sum()
            else:
                lost_superior = (z_vals < vox_min[z_axis]).sum()
                lost_inferior = (z_vals > vox_max[z_axis]).sum()
            lost_xy = n_lost - lost_superior - lost_inferior
        else:
            lost_superior = lost_inferior = lost_xy = 0

        rows.append({
            "filename": fname, "extent_z_mm": extent_z, "cropped": True,
            "n_gt_voxels": n_gt, "n_lost": n_lost, "frac_lost": frac_lost,
            "frac_lost_inferior": lost_inferior / n_gt,
            "frac_lost_superior": lost_superior / n_gt,
            "frac_lost_xy": lost_xy / n_gt,
        })
        flag = "  <-- LOSING GT SIGNAL" if frac_lost > 0.01 else ""
        print(f"{fname}: GT={n_gt} voxels, lost={n_lost} ({frac_lost:.1%}) "
              f"[inferior={lost_inferior/n_gt:.1%} superior={lost_superior/n_gt:.1%} "
              f"xy={lost_xy/n_gt:.1%}]{flag}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    cropped = df[df["cropped"]]
    print(f"\n=== SUMMARY (n={len(df)} GT cases, {len(cropped)} ROI-cropped) ===")
    print(f"Cases with ANY GT voxels lost to the crop: {(cropped['frac_lost'] > 0).sum()} / {len(cropped)}")
    print(f"Cases losing >1% of GT voxels: {(cropped['frac_lost'] > 0.01).sum()} / {len(cropped)}")
    print(f"Cases losing >10% of GT voxels: {(cropped['frac_lost'] > 0.10).sum()} / {len(cropped)}")
    print(f"Mean frac_lost: {cropped['frac_lost'].mean():.4f}")
    print(f"Mean frac_lost_inferior: {cropped['frac_lost_inferior'].mean():.4f}")
    print(f"Mean frac_lost_superior: {cropped['frac_lost_superior'].mean():.4f}")
    print(f"\nSaved per-case results to {OUT_CSV}")


if __name__ == "__main__":
    main()
