"""Render our ROI pipeline's bbox (red rectangle) + GT free-air mask (green)
overlaid on the native CT at a specific slice, for one case -- direct visual
answer to "is the bbox cutting off real signal at this slice."
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = "20251028056-1.nii.gz"
CT_PATH = os.path.join(ROOT, "merged", "Pneumo_Positive", CASE)
GT_PATH = os.path.join(ROOT, "merged", "GT", CASE)
OUT_PATH = os.path.join(ROOT, "scripts", "air_mask_qc", f"roi_bbox_result__{CASE.replace('.nii.gz','')}.png")

BBOX_MIN_MM = (-245.840473, -270.93768, -502.214077)
BBOX_MAX_MM = (235.264814, 61.542021, -222.214077)
Z_SLICE = 94

ct_img = nib.as_closest_canonical(nib.load(CT_PATH))
gt_img = nib.as_closest_canonical(nib.load(GT_PATH))
ct = ct_img.get_fdata(dtype=np.float32)
gt = gt_img.get_fdata().astype(bool)
affine = ct_img.affine
shape = ct.shape

inv = np.linalg.inv(affine)
corners_mm = np.array([
    [BBOX_MIN_MM[0], BBOX_MIN_MM[1], BBOX_MIN_MM[2], 1],
    [BBOX_MAX_MM[0], BBOX_MAX_MM[1], BBOX_MAX_MM[2], 1],
])
corners_vox = (inv @ corners_mm.T).T[:, :3]
vox_min = np.floor(corners_vox.min(axis=0)).astype(int)
vox_max = np.ceil(corners_vox.max(axis=0)).astype(int)
vox_min_c = np.clip(vox_min, 0, np.array(shape) - 1)
vox_max_c = np.clip(vox_max, 0, np.array(shape) - 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# --- axial slice at z=94 ---
ct_slice = np.clip(ct[:, :, Z_SLICE], -200, 200).T
gt_slice = gt[:, :, Z_SLICE].T
axes[0].imshow(ct_slice, cmap="gray", origin="lower")
gt_overlay = np.zeros((*gt_slice.shape, 4))
gt_overlay[gt_slice] = [0, 1, 0, 0.5]
axes[0].imshow(gt_overlay, origin="lower")
rect = patches.Rectangle((vox_min_c[0], vox_min_c[1]), vox_max_c[0] - vox_min_c[0], vox_max_c[1] - vox_min_c[1],
                          linewidth=2, edgecolor="red", facecolor="none")
axes[0].add_patch(rect)
n_gt_this_slice = int(gt_slice.sum())
axes[0].set_title(f"axial z={Z_SLICE} | GT voxels this slice={n_gt_this_slice} (green)\nred=our ROI bbox (x/y)")
axes[0].axis("off")

# --- sagittal view through the bbox x-center, showing z-extent + GT + bbox ---
x_mid = (vox_min_c[0] + vox_max_c[0]) // 2
ct_sag = np.clip(ct[x_mid, :, :], -200, 200).T
gt_sag = gt[x_mid, :, :].T
axes[1].imshow(ct_sag, cmap="gray", origin="lower")
gt_overlay_sag = np.zeros((*gt_sag.shape, 4))
gt_overlay_sag[gt_sag] = [0, 1, 0, 0.5]
axes[1].imshow(gt_overlay_sag, origin="lower")
rect2 = patches.Rectangle((vox_min_c[1], vox_min_c[2]), vox_max_c[1] - vox_min_c[1], vox_max_c[2] - vox_min_c[2],
                           linewidth=2, edgecolor="red", facecolor="none")
axes[1].add_patch(rect2)
axes[1].axhline(Z_SLICE, color="cyan", linestyle="--", linewidth=1)
n_gt_total = int(gt.sum())
axes[1].set_title(f"sagittal x={x_mid} | GT total voxels={n_gt_total} (green)\nred=our ROI bbox (y/z), cyan=z={Z_SLICE}")
axes[1].axis("off")

fig.suptitle(f"{CASE}: our VISTA3D ROI pipeline's bbox vs. real GT free-air mask")
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=120)
print(f"Saved {OUT_PATH}")
