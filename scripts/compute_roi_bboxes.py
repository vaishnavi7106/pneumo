"""
Batch-compute (and cache) ROI bounding boxes for all extended-FOV volumes,
so the dataset/dataloader doesn't have to re-run VISTA3D segmentation on
every epoch. Includes a spot-check mode that saves middle-slice PNGs with
the bbox overlaid for a handful of volumes, to visually verify before
trusting the full batch (258 volumes is too many to check by hand).
"""
import csv
import os
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roi_localizer import ROI_ORGAN_IDS, segment_organ_bbox_mm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
OUT_CSV = os.path.join(ROOT, "scripts", "roi_bboxes.csv")
SPOTCHECK_DIR = os.path.join(ROOT, "scripts", "spotcheck_png")


def mm_to_vox_bbox(filepath, bbox_min_mm, bbox_max_mm):
    """Convert a physical-mm bbox back to voxel index bounds in the volume's
    OWN on-disk RAS-reoriented grid (for spot-check plotting only)."""
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


def spot_check_plot(filepath, bbox_min_mm, bbox_max_mm, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    img, vox_min, vox_max = mm_to_vox_bbox(filepath, bbox_min_mm, bbox_max_mm)
    data = img.get_fdata(dtype=np.float32)
    shape = data.shape

    vox_min_c = np.clip(vox_min, 0, np.array(shape) - 1)
    vox_max_c = np.clip(vox_max, 0, np.array(shape) - 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # axial (z mid of bbox), coronal (y mid), sagittal (x mid) -- through bbox center
    center = ((vox_min_c + vox_max_c) // 2).astype(int)

    # axial slice (X-Y plane at center z)
    z = int(np.clip(center[2], 0, shape[2] - 1))
    axes[0].imshow(data[:, :, z].T, cmap="gray", origin="lower", vmin=-200, vmax=400)
    rect = patches.Rectangle((vox_min_c[0], vox_min_c[1]), vox_max_c[0] - vox_min_c[0],
                              vox_max_c[1] - vox_min_c[1], linewidth=2, edgecolor="lime", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].set_title(f"axial z={z}")

    # coronal slice (X-Z plane at center y)
    y = int(np.clip(center[1], 0, shape[1] - 1))
    axes[1].imshow(data[:, y, :].T, cmap="gray", origin="lower", vmin=-200, vmax=400)
    rect = patches.Rectangle((vox_min_c[0], vox_min_c[2]), vox_max_c[0] - vox_min_c[0],
                              vox_max_c[2] - vox_min_c[2], linewidth=2, edgecolor="lime", facecolor="none")
    axes[1].add_patch(rect)
    axes[1].set_title(f"coronal y={y}")

    # sagittal slice (Y-Z plane at center x)
    x = int(np.clip(center[0], 0, shape[0] - 1))
    axes[2].imshow(data[x, :, :].T, cmap="gray", origin="lower", vmin=-200, vmax=400)
    rect = patches.Rectangle((vox_min_c[1], vox_min_c[2]), vox_max_c[1] - vox_min_c[1],
                              vox_max_c[2] - vox_min_c[2], linewidth=2, edgecolor="lime", facecolor="none")
    axes[2].add_patch(rect)
    axes[2].set_title(f"sagittal x={x}")

    fig.suptitle(os.path.basename(filepath))
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main(spot_check_n: int = 5, full_batch: bool = True):
    df = pd.read_csv(AUDIT_CSV)
    extended = df[df["extent_z_mm"] >= 375].reset_index(drop=True)
    print(f"{len(extended)} extended-FOV volumes to process")

    os.makedirs(SPOTCHECK_DIR, exist_ok=True)

    # spot-check a stratified handful first: mix of positive/negative + one thick-slice
    spot_rows = pd.concat([
        extended[extended.label == "positive"].sample(min(2, (extended.label == "positive").sum()), random_state=0),
        extended[extended.label == "negative"].sample(min(2, (extended.label == "negative").sum()), random_state=0),
        extended[extended.spacing_z > 10].head(1),
    ]).drop_duplicates(subset="filepath")
    print(f"\n=== SPOT CHECK: {len(spot_rows)} volumes ===")
    spot_results = []
    for _, row in spot_rows.iterrows():
        t0 = time.time()
        r = segment_organ_bbox_mm(row.filepath)
        elapsed = time.time() - t0
        n_found = len(r["found_labels"])
        print(f"{row.filename}: status={r['status']}, found {n_found}/{len(ROI_ORGAN_IDS)} organs, "
              f"{elapsed:.1f}s")
        if r["status"] == "ok":
            png_path = os.path.join(SPOTCHECK_DIR, row.filename.replace(".nii.gz", ".png"))
            spot_check_plot(row.filepath, r["bbox_min_mm"], r["bbox_max_mm"], png_path)
            print(f"  saved {png_path}")
        spot_results.append((row.filename, r))

    if not full_batch:
        return spot_results

    print(f"\n=== FULL BATCH: {len(extended)} volumes ===")
    fieldnames = ["filepath", "filename", "status", "n_found_organs", "found_labels", "elapsed_s",
                  "bbox_min_x", "bbox_min_y", "bbox_min_z", "bbox_max_x", "bbox_max_y", "bbox_max_z", "error"]
    records = []
    # write incrementally so a crash/interrupt near the end doesn't lose all prior work
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(extended.itertuples(index=False), total=len(extended), desc="ROI bbox", file=sys.stdout)
        for row in pbar:
            t0 = time.time()
            try:
                r = segment_organ_bbox_mm(row.filepath)
            except Exception as e:  # noqa: BLE001
                pbar.write(f"{row.filename}: ERROR {type(e).__name__}: {e}")
                rec = {"filepath": row.filepath, "filename": row.filename, "status": "error", "error": repr(e)}
                writer.writerow(rec)
                f.flush()
                records.append(rec)
                continue
            elapsed = time.time() - t0
            rec = {
                "filepath": row.filepath, "filename": row.filename, "status": r["status"],
                "n_found_organs": len(r["found_labels"]),
                "found_labels": ",".join(map(str, sorted(r["found_labels"]))),
                "elapsed_s": elapsed,
            }
            if r["status"] == "ok":
                rec["bbox_min_x"], rec["bbox_min_y"], rec["bbox_min_z"] = r["bbox_min_mm"]
                rec["bbox_max_x"], rec["bbox_max_y"], rec["bbox_max_z"] = r["bbox_max_mm"]
            writer.writerow(rec)
            f.flush()
            records.append(rec)
            pbar.set_postfix(status=r["status"], organs=len(r["found_labels"]), s=f"{elapsed:.1f}")

    out_df = pd.DataFrame(records)
    print(f"\nSaved ROI bboxes to {OUT_CSV}")
    print(out_df["status"].value_counts())
    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot-check-only", action="store_true")
    args = parser.parse_args()
    main(full_batch=not args.spot_check_only)
