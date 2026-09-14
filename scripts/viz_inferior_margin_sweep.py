"""
Visualize how much of the real GT free-air signal gets recovered as the
INFERIOR z-margin (currently a fixed 100mm below the core-organ centroid,
see roi_localizer.py/compute_fixed_z_window.py) is extended further down by
5/10/15/20mm -- sagittal view, GT overlay, one horizontal line per margin.

Current lower bound has NOTHING to do with bone -- it's core_organ_centroid_z
minus a fixed 100mm constant (see CORE_ORGAN_IDS_FOR_CENTROID in
roi_localizer.py). This sweep answers: does a bit more margin actually
recover the missed inferior GT voxels, or is the loss happening further down
than even a modest extension would reach?
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = "20251028056-1.nii.gz"
CT_PATH = os.path.join(ROOT, "merged", "Pneumo_Positive", CASE)
GT_PATH = os.path.join(ROOT, "merged", "GT", CASE)
OUT_PATH = os.path.join(ROOT, "scripts", "air_mask_qc", f"inferior_margin_sweep__{CASE.replace('.nii.gz','')}.png")

# from roi_bboxes.csv for this case (already computed via our pipeline)
CORE_CENTROID_Z_MM = -402.214077
CURRENT_INFERIOR_MARGIN_MM = 100.0
EXTRA_MARGINS_MM = [0, 5, 10, 15, 20]  # 0 = current bound, others = extended further inferior
BBOX_MIN_X_MM, BBOX_MAX_X_MM = -245.840473, 235.264814
BBOX_MIN_Y_MM, BBOX_MAX_Y_MM = -270.93768, 61.542021
BBOX_MAX_Z_MM = -222.214077  # superior bound unchanged

ct_img = nib.as_closest_canonical(nib.load(CT_PATH))
gt_img = nib.as_closest_canonical(nib.load(GT_PATH))
ct = ct_img.get_fdata(dtype=np.float32)
gt = gt_img.get_fdata().astype(bool)
affine = ct_img.affine
shape = ct.shape
inv = np.linalg.inv(affine)


def mm_to_vox(x_mm, y_mm, z_mm):
    v = inv @ np.array([x_mm, y_mm, z_mm, 1])
    return v[:3]

x_mid_mm = (BBOX_MIN_X_MM + BBOX_MAX_X_MM) / 2
current_min_z_mm = CORE_CENTROID_Z_MM - CURRENT_INFERIOR_MARGIN_MM

x_mid_vox = mm_to_vox(x_mid_mm, 0, 0)[0]
x_mid = int(np.clip(round(x_mid_vox), 0, shape[0] - 1))

ct_sag = np.clip(ct[x_mid, :, :], -200, 200).T
gt_sag = gt[x_mid, :, :].T

fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(ct_sag, cmap="gray", origin="lower")
gt_overlay = np.zeros((*gt_sag.shape, 4))
gt_overlay[gt_sag] = [0, 1, 0, 0.55]
ax.imshow(gt_overlay, origin="lower")

n_gt_total = int(gt.sum())
colors = ["red", "orange", "gold", "cyan", "deepskyblue"]
lines_info = []
for extra_mm, color in zip(EXTRA_MARGINS_MM, colors):
    z_mm = current_min_z_mm - extra_mm
    z_vox = mm_to_vox(x_mid_mm, (BBOX_MIN_Y_MM + BBOX_MAX_Y_MM) / 2, z_mm)[2]
    z_vox = int(round(z_vox))
    z_vox_c = int(np.clip(z_vox, 0, shape[2] - 1))
    ax.axhline(z_vox_c, color=color, linewidth=2,
               label=f"{'current bound' if extra_mm == 0 else f'+{extra_mm}mm lower'} (z_vox={z_vox_c})")
    # GT voxels between this line and the NEXT (smaller) margin's line -- recovered by extending to here
    n_below_this = int((gt[:, :, :z_vox_c]).sum()) if z_vox_c > 0 else 0
    lines_info.append((extra_mm, z_vox_c, n_below_this))

ax.legend(loc="upper right", fontsize=9)
ax.set_title(f"{CASE}  sagittal x={x_mid}  |  total GT voxels={n_gt_total}\n"
             f"lines = inferior z-bound at current + extra margin (mm)")
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=120)
print(f"Saved {OUT_PATH}\n")

print(f"{'extra_mm':>10} {'z_vox':>8} {'GT voxels still below this line (whole volume)':>50}")
for extra_mm, z_vox_c, n_below in lines_info:
    print(f"{extra_mm:>10} {z_vox_c:>8} {n_below:>50}")
