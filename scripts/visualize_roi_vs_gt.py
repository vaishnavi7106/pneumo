"""
Visual QC: overlay the GT free-air mask and the cached ROI bbox on the
native-resolution CT, for a specific case -- to see exactly what's being
cropped off, not just the aggregate percentage.
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
GT_DIR = os.path.join(ROOT, "merged", "GT")
CT_DIR = os.path.join(ROOT, "merged", "Pneumo_Positive")
ROI_CSV = os.path.join(ROOT, "scripts", "roi_bboxes.csv")
OUT_DIR = os.path.join(ROOT, "scripts", "roi_vs_gt_qc")


def mm_bbox_to_vox(affine, bbox_min_mm, bbox_max_mm):
    inv = np.linalg.inv(affine)
    corners_mm = np.array([
        [bbox_min_mm[0], bbox_min_mm[1], bbox_min_mm[2], 1],
        [bbox_max_mm[0], bbox_max_mm[1], bbox_max_mm[2], 1],
    ])
    corners_vox = (inv @ corners_mm.T).T[:, :3]
    return corners_vox.min(axis=0), corners_vox.max(axis=0)


def main(fnames):
    os.makedirs(OUT_DIR, exist_ok=True)
    roi = pd.read_csv(ROI_CSV)
    roi["basename"] = roi["filepath"].apply(os.path.basename)
    roi_ok = roi[roi["status"] == "ok"].set_index("basename")

    for fname in fnames:
        ct_path = os.path.join(CT_DIR, fname)
        gt_path = os.path.join(GT_DIR, fname)
        ct_img = nib.load(ct_path)
        gt_img = nib.load(gt_path)
        ct = ct_img.get_fdata(dtype=np.float32)
        gt = gt_img.get_fdata().astype(bool)
        affine = ct_img.affine

        r = roi_ok.loc[fname]
        bbox_min_mm = (r["bbox_min_x"], r["bbox_min_y"], r["bbox_min_z"])
        bbox_max_mm = (r["bbox_max_x"], r["bbox_max_y"], r["bbox_max_z"])
        vox_min, vox_max = mm_bbox_to_vox(affine, bbox_min_mm, bbox_max_mm)

        s_row = np.abs(affine[2, :3])
        z_axis = int(np.argmax(s_row))

        # normalize CT display range
        disp = np.clip(ct, -200, 400)
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)

        gt_coords = np.argwhere(gt)
        z_center = int(np.median(gt_coords[:, z_axis]))  # slice through the bulk of the GT mask
        y_center = ct.shape[1] // 2
        x_center = ct.shape[0] // 2

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        # sagittal (fix x), coronal (fix y), axial (fix z_axis) -- show GT (green)
        # and bbox extent (red rectangle) on each
        def bbox_rect_for(dim_a, dim_b):
            lo = (vox_min[dim_a], vox_min[dim_b])
            w = vox_max[dim_a] - vox_min[dim_a]
            h = vox_max[dim_b] - vox_min[dim_b]
            return lo, w, h

        # sagittal: fix axis 0, show axes (1,2)
        ax = axes[0]
        sl = disp[x_center, :, :].T
        gtsl = gt[x_center, :, :].T
        ax.imshow(sl, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(~gtsl, gtsl), cmap="Greens", origin="lower", alpha=0.6)
        (lo0, lo1), w, h = bbox_rect_for(1, 2)
        ax.add_patch(patches.Rectangle((lo0, lo1), w, h, fill=False, edgecolor="red", linewidth=1.5))
        ax.set_title("sagittal -- GT (green) vs ROI bbox (red)")
        ax.axis("off")

        # coronal: fix axis 1, show axes (0,2)
        ax = axes[1]
        sl = disp[:, y_center, :].T
        gtsl = gt[:, y_center, :].T
        ax.imshow(sl, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(~gtsl, gtsl), cmap="Greens", origin="lower", alpha=0.6)
        (lo0, lo1), w, h = bbox_rect_for(0, 2)
        ax.add_patch(patches.Rectangle((lo0, lo1), w, h, fill=False, edgecolor="red", linewidth=1.5))
        ax.set_title("coronal -- GT (green) vs ROI bbox (red)")
        ax.axis("off")

        # axial through GT bulk
        ax = axes[2]
        z_idx = np.clip(z_center, 0, ct.shape[2] - 1) if z_axis == 2 else ct.shape[2] // 2
        sl = disp[:, :, z_idx].T
        gtsl = gt[:, :, z_idx].T
        ax.imshow(sl, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(~gtsl, gtsl), cmap="Greens", origin="lower", alpha=0.6)
        (lo0, lo1), w, h = bbox_rect_for(0, 1)
        ax.add_patch(patches.Rectangle((lo0, lo1), w, h, fill=False, edgecolor="red", linewidth=1.5))
        ax.set_title(f"axial (through GT bulk, z={z_idx}) -- GT (green) vs ROI bbox (red)")
        ax.axis("off")

        frac_outside = 1.0 - (
            np.all((gt_coords >= np.floor(vox_min)) & (gt_coords <= np.ceil(vox_max)), axis=1).mean()
        )
        fig.suptitle(f"{fname} -- {frac_outside:.1%} of GT free-air voxels fall OUTSIDE the ROI crop")
        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f"roi_vs_gt__{fname.replace('.nii.gz', '')}.png")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"{fname}: {frac_outside:.1%} outside bbox -> saved {out_path}")


if __name__ == "__main__":
    cases = sys.argv[1:] if len(sys.argv) > 1 else [
        "20251028013-1.nii.gz",  # worst case, 67.7% lost
        "20251028066-1.nii.gz",  # 60.7% lost
    ]
    main(cases)
