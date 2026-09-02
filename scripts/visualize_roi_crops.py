"""
Visualize ROI-cropped cases for investigation: for a sample of volumes, shows
(1) the original volume with the cached ROI bbox overlaid (physical-mm bbox
converted back to that volume's own voxel grid), and (2) the ACTUAL final
preprocessed tensor -- post reorient/resample/crop/intensity-scale/resize --
i.e. literally what VistaClassifier receives as input. (1) answers "did the
bbox find the right region"; (2) answers "what does the network actually see
after the crude resize to patch_size", which is the more revealing one for
spotting resize distortion, over/under-cropping, or intensity issues.

Runs entirely on CPU (no VISTA3D forward pass, no GPU) -- safe to run
alongside an active GPU training job.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (  # noqa: E402
    EXTENDED_FOV_THRESHOLD_MM, PneumoDataset, ROI_BBOX_CSV,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
OUT_DIR = os.path.join(ROOT, "scripts", "roi_investigation_png")


def mm_to_vox_bbox(filepath, bbox_min_mm, bbox_max_mm):
    img = nib.as_closest_canonical(nib.load(filepath))
    affine = img.affine
    inv = np.linalg.inv(affine)
    corners_mm = np.array([
        [bbox_min_mm[0], bbox_min_mm[1], bbox_min_mm[2], 1],
        [bbox_max_mm[0], bbox_max_mm[1], bbox_max_mm[2], 1],
    ])
    corners_vox = (inv @ corners_mm.T).T[:, :3]
    vox_min = np.floor(corners_vox.min(axis=0)).astype(int)
    vox_max = np.ceil(corners_vox.max(axis=0)).astype(int)
    return img, vox_min, vox_max


def plot_case(row, roi_row, ds: PneumoDataset, out_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    is_extended = row["extent_z_mm"] >= EXTENDED_FOV_THRESHOLD_MM

    # --- row 1: original volume with bbox overlay (if extended-FOV) ---
    img = nib.as_closest_canonical(nib.load(row["filepath"]))
    data = img.get_fdata(dtype=np.float32)
    shape = data.shape
    voxel_mm = nib.affines.voxel_sizes(img.affine)  # (R,A,S) mm per voxel -- can be
    # highly anisotropic (e.g. 0.6x5.0x0.6mm) for some orientation families; imshow
    # must use aspect=mm_ratio or a thin-but-physically-normal axis renders as a
    # squished strip that looks like a bug but isn't one

    if is_extended and roi_row is not None:
        bbox_min_mm = (roi_row["bbox_min_x"], roi_row["bbox_min_y"], roi_row["bbox_min_z"])
        bbox_max_mm = (roi_row["bbox_max_x"], roi_row["bbox_max_y"], roi_row["bbox_max_z"])
        _, vox_min, vox_max = mm_to_vox_bbox(row["filepath"], bbox_min_mm, bbox_max_mm)
        vox_min_c = np.clip(vox_min, 0, np.array(shape) - 1)
        vox_max_c = np.clip(vox_max, 0, np.array(shape) - 1)
        center = ((vox_min_c + vox_max_c) // 2).astype(int)
    else:
        vox_min_c = np.array([0, 0, 0])
        vox_max_c = np.array(shape) - 1
        center = (np.array(shape) // 2).astype(int)

    for ax, (dim, other1, other2, label) in zip(
        axes[0], [(2, 0, 1, "axial"), (1, 0, 2, "coronal"), (0, 1, 2, "sagittal")]
    ):
        idx = int(np.clip(center[dim], 0, shape[dim] - 1))
        if dim == 2:
            sl = data[:, :, idx].T
        elif dim == 1:
            sl = data[:, idx, :].T
        else:
            sl = data[idx, :, :].T
        # aspect = (mm per voxel along the plotted vertical axis) / (mm per voxel
        # along the plotted horizontal axis) -- makes anisotropic spacing render
        # in correct physical proportion instead of raw (and misleading) pixel count
        aspect = voxel_mm[other2] / voxel_mm[other1]
        ax.imshow(sl, cmap="gray", origin="lower", vmin=-200, vmax=400, aspect=aspect)
        if is_extended and roi_row is not None:
            o1lo, o1hi = vox_min_c[other1], vox_max_c[other1]
            o2lo, o2hi = vox_min_c[other2], vox_max_c[other2]
            rect = patches.Rectangle((o1lo, o2lo), o1hi - o1lo, o2hi - o2lo,
                                      linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
        ax.set_title(f"ORIGINAL {label} (idx={idx})")

    # --- row 2: actual preprocessed tensor fed to the model ---
    idx_in_ds = ds.filepaths.index(row["filepath"])
    x, label_t, _ = ds[idx_in_ds]
    x = x.squeeze(0).numpy()  # [D, H, W] in [0,1]
    pshape = x.shape
    pcenter = np.array(pshape) // 2

    for ax, (dim, label) in zip(axes[1], [(2, "axial"), (1, "coronal"), (0, "sagittal")]):
        idx = int(pcenter[dim])
        if dim == 2:
            sl = x[:, :, idx].T
        elif dim == 1:
            sl = x[:, idx, :].T
        else:
            sl = x[idx, :, :].T
        ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(f"PREPROCESSED {label} (idx={idx}, shape={pshape})")

    fov_type = "extended-FOV (cropped)" if is_extended else "abdomen-only (no crop)"
    fig.suptitle(f"{row['filename']}  |  label={row['label']}  |  {fov_type}  |  "
                 f"extent_z={row['extent_z_mm']:.0f}mm  spacing_z={row['spacing_z']:.1f}mm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main(n_per_group: int = 3, patch_size: int = 224, seed: int = 0):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(AUDIT_CSV)
    roi_df = pd.read_csv(ROI_BBOX_CSV) if os.path.exists(ROI_BBOX_CSV) else None

    rng = np.random.RandomState(seed)

    extended_pos = df[(df.extent_z_mm >= EXTENDED_FOV_THRESHOLD_MM) & (df.label == "positive")]
    extended_neg = df[(df.extent_z_mm >= EXTENDED_FOV_THRESHOLD_MM) & (df.label == "negative")]
    abdomen_pos = df[(df.extent_z_mm < EXTENDED_FOV_THRESHOLD_MM) & (df.label == "positive")]
    abdomen_neg = df[(df.extent_z_mm < EXTENDED_FOV_THRESHOLD_MM) & (df.label == "negative")]
    thick = df[df.spacing_z > 10]

    groups = {
        "extended_positive": extended_pos.sample(min(n_per_group, len(extended_pos)), random_state=seed),
        "extended_negative": extended_neg.sample(min(n_per_group, len(extended_neg)), random_state=seed),
        "abdomen_positive": abdomen_pos.sample(min(n_per_group, len(abdomen_pos)), random_state=seed),
        "abdomen_negative": abdomen_neg.sample(min(n_per_group, len(abdomen_neg)), random_state=seed),
        "thick_slice_outliers": thick.sample(min(n_per_group, len(thick)), random_state=seed),
    }
    all_rows = pd.concat(groups.values()).drop_duplicates(subset="filepath")
    print(f"Selected {len(all_rows)} volumes across groups: "
          f"{ {k: len(v) for k, v in groups.items()} }")

    ds = PneumoDataset(all_rows["filepath"].tolist(), patch_size=(patch_size,) * 3)

    for group_name, rows in groups.items():
        for _, row in rows.iterrows():
            roi_row = None
            if roi_df is not None:
                match = roi_df[roi_df["filepath"] == row["filepath"]]
                if len(match) and match.iloc[0].get("status") == "ok":
                    roi_row = match.iloc[0]
            out_path = os.path.join(OUT_DIR, f"{group_name}__{row['filename'].replace('.nii.gz', '')}.png")
            try:
                plot_case(row, roi_row, ds, out_path)
                print(f"[{group_name}] {row['filename']}: saved {out_path}")
            except Exception as e:  # noqa: BLE001
                print(f"[{group_name}] {row['filename']}: ERROR {type(e).__name__}: {e}")

    print(f"\nAll images saved to {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-group", type=int, default=3)
    p.add_argument("--patch-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(n_per_group=args.n_per_group, patch_size=args.patch_size, seed=args.seed)
