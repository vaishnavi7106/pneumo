"""
Check whether VISTA3D's intensity window (HU_A_MIN..HU_A_MAX, used for [0,1]
normalization) clips or dilutes real free-air signal, using the 23 GT
free-air segmentations in merged/GT/ as ground truth. Reports the actual HU
distribution of GT-positive (free-air) voxels, how much of that distribution
falls below HU_A_MIN (loses raw contrast to clipping), and how much of the
final [0,1] normalized range true air voxels actually occupy (dilution).
"""
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(ROOT, "merged", "GT")
CT_DIR = os.path.join(ROOT, "merged", "Pneumo_Positive")

HU_A_MIN = -963.8247715525971
HU_A_MAX = 1053.678477684517


def main():
    gt_files = sorted(os.listdir(GT_DIR))
    all_hu = []
    n_clipped_low = 0
    n_total = 0
    per_case = []

    for fname in gt_files:
        gt_path = os.path.join(GT_DIR, fname)
        ct_path = os.path.join(CT_DIR, fname)
        if not os.path.exists(ct_path):
            print(f"SKIP {fname}: no matching CT in {CT_DIR}")
            continue

        gt_img = nib.as_closest_canonical(nib.load(gt_path))
        ct_img = nib.as_closest_canonical(nib.load(ct_path))
        gt = gt_img.get_fdata(dtype=np.float32) > 0.5
        hu = ct_img.get_fdata(dtype=np.float32)

        if gt.shape != hu.shape:
            print(f"SKIP {fname}: shape mismatch gt={gt.shape} ct={hu.shape}")
            continue

        air_hu = hu[gt]
        if air_hu.size == 0:
            print(f"SKIP {fname}: empty GT mask")
            continue

        clipped = (air_hu < HU_A_MIN).sum()
        n_clipped_low += clipped
        n_total += air_hu.size
        all_hu.append(air_hu)
        per_case.append((fname, air_hu.size, clipped, air_hu.min(), np.percentile(air_hu, 1),
                          np.percentile(air_hu, 50), np.percentile(air_hu, 99), air_hu.max()))

    all_hu = np.concatenate(all_hu)
    print(f"\n=== GT free-air voxel HU distribution (n={all_hu.size} voxels, {len(per_case)} cases) ===")
    print(f"min={all_hu.min():.1f}  p1={np.percentile(all_hu,1):.1f}  p5={np.percentile(all_hu,5):.1f}  "
          f"p50={np.percentile(all_hu,50):.1f}  p95={np.percentile(all_hu,95):.1f}  "
          f"p99={np.percentile(all_hu,99):.1f}  max={all_hu.max():.1f}")
    print(f"\nHU_A_MIN={HU_A_MIN:.1f}, HU_A_MAX={HU_A_MAX:.1f} (VISTA3D window, width={HU_A_MAX-HU_A_MIN:.1f})")
    print(f"Voxels clipped below HU_A_MIN: {n_clipped_low}/{n_total} ({100*n_clipped_low/n_total:.3f}%)")

    normed = np.clip((all_hu - HU_A_MIN) / (HU_A_MAX - HU_A_MIN), 0, 1)
    print(f"\nNormalized [0,1] range occupied by true free-air voxels:")
    print(f"  min={normed.min():.4f}  p1={np.percentile(normed,1):.4f}  p50={np.percentile(normed,50):.4f}  "
          f"p99={np.percentile(normed,99):.4f}  max={normed.max():.4f}")
    print(f"  => air voxels occupy roughly [{np.percentile(normed,1):.3f}, {np.percentile(normed,99):.3f}] "
          f"of the full [0,1] normalized channel (98% of air mass)")

    print(f"\n=== per-case breakdown ===")
    print(f"{'file':<25} {'n_vox':>8} {'clipped':>8} {'min':>8} {'p1':>8} {'p50':>8} {'p99':>8} {'max':>8}")
    for fname, n, clipped, mn, p1, p50, p99, mx in per_case:
        print(f"{fname:<25} {n:>8} {clipped:>8} {mn:>8.1f} {p1:>8.1f} {p50:>8.1f} {p99:>8.1f} {mx:>8.1f}")


if __name__ == "__main__":
    main()
