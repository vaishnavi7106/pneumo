"""
Validate adaptive_roi_crop.py's 3-axis tight bbox against all 23 real GT
free-air segmentations: does the crop ever clip real signal (fraction of GT
voxels falling outside the bbox), and how much volume reduction does it
achieve -- directly comparable to roi_vs_gt_coverage.csv (our current
fixed-window approach's numbers from check_roi_vs_gt.py).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_roi_crop import compute_final_bbox, load_ras_native  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(ROOT, "merged", "GT")
CT_DIR = os.path.join(ROOT, "merged", "Pneumo_Positive")
OUT_CSV = os.path.join(ROOT, "scripts", "adaptive_roi_vs_gt_coverage.csv")


def main():
    gt_files = sorted(os.listdir(GT_DIR))
    rows = []

    for fname in gt_files:
        ct_path = os.path.join(CT_DIR, fname)
        gt_path = os.path.join(GT_DIR, fname)
        if not os.path.exists(ct_path):
            print(f"SKIP {fname}: no matching CT")
            continue

        import nibabel as nib
        gt = nib.as_closest_canonical(nib.load(gt_path)).get_fdata().astype(bool)
        n_gt = int(gt.sum())
        if n_gt == 0:
            print(f"SKIP {fname}: empty GT mask")
            continue

        hu_array, affine, zooms = load_ras_native(ct_path)
        assert hu_array.shape == gt.shape, (fname, hu_array.shape, gt.shape)

        bbox, uncapped_bbox, mask, diag = compute_final_bbox(hu_array, zooms, affine)

        inside = np.zeros_like(gt)
        inside[bbox] = True
        n_lost = int((gt & ~inside).sum())
        frac_lost = n_lost / n_gt

        vol_total = hu_array.size
        vol_cropped = int(np.prod([s.stop - s.start for s in bbox]))
        frac_reduction = 1.0 - (vol_cropped / vol_total)
        was_capped = diag["z_cap"]["capped"]

        rows.append({
            "filename": fname, "n_gt_voxels": n_gt, "n_lost": n_lost, "frac_lost": frac_lost,
            "volume_reduction": frac_reduction, "z_capped": was_capped,
            "uncapped_z_extent_mm": diag["z_cap"]["extent_mm"],
        })
        flag = "  <-- LOSING GT SIGNAL" if frac_lost > 0.001 else ""
        cap_note = " [z-capped]" if was_capped else ""
        print(f"{fname}: GT={n_gt}, lost={n_lost} ({frac_lost:.2%}), "
              f"volume_reduction={frac_reduction:.1%}{cap_note}{flag}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n=== SUMMARY (n={len(df)}) ===")
    print(f"Cases with ANY GT voxels lost: {(df['frac_lost'] > 0).sum()} / {len(df)}")
    print(f"Mean frac_lost: {df['frac_lost'].mean():.4%}")
    print(f"Max frac_lost: {df['frac_lost'].max():.4%}")
    print(f"Mean volume reduction: {df['volume_reduction'].mean():.1%}")
    print(f"Min volume reduction: {df['volume_reduction'].min():.1%}")
    print(f"Cases where the z-cap kicked in: {df['z_capped'].sum()} / {len(df)}")
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
