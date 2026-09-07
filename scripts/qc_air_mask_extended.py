"""
Extended visual QC for the air-mask channel: instead of aggregate stats over
a handful of volumes, this scans ALL positive volumes for post-fix air-mask
extent and renders full QC panels for a larger, more varied sample --
specifically including the positives with the LARGEST air-mask volume, since
a big and/or more central free-air collection is the case most likely to
stress-test the 3-voxel closing kernel differently than the small bowel-gas
pockets the fix was tuned against (more wall-contact surface area for the
closing to interact with; more room for the erode-back step to visibly eat
into a large collection's true extent rather than just trimming a 1-voxel
skin-surface sliver).

Also reports a "centrality" score per case: the max distance (mm) from any
air-mask voxel to the nearest point OUTSIDE the body silhouette -- larger
means the air reaches deeper into the body rather than just hugging the
abdominal wall (a purely wall-hugging collection would score near 0).
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import AUDIT_CSV, RESAMPLE_SPACING, ROI_BBOX_CSV, _compute_body_mask  # noqa: E402
from diagnose_air_mask_fix import load_raw_hu  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "scripts", "air_mask_qc_extended")
AIR_THRESHOLD = -600.0
N_TOP_POSITIVE = 6
N_RANDOM_POSITIVE = 4
N_RANDOM_NEGATIVE = 4
VOXEL_MM = RESAMPLE_SPACING[0]


def compute_air_and_body(filepath, audit_df, roi_df):
    arr = load_raw_hu(filepath, audit_df, roi_df)
    body_mask = _compute_body_mask(torch.as_tensor(arr)).numpy().astype(bool)
    air_mask = (arr < AIR_THRESHOLD) & body_mask
    return arr, body_mask, air_mask


def centrality_score(body_mask: np.ndarray, air_mask: np.ndarray) -> float:
    if not air_mask.any():
        return 0.0
    dist_to_outside = ndimage.distance_transform_edt(body_mask, sampling=(VOXEL_MM,) * 3)
    return float(dist_to_outside[air_mask].max())


def render_panel(arr, air_mask, title, out_path):
    intensity = np.clip((arr + 963.82) / (1053.68 + 963.82), 0, 1)
    c = [s // 2 for s in arr.shape]
    if air_mask.any():
        # center the slices on the air mask's centroid (not volume center) so
        # a large-but-off-center collection is actually visible in the panel
        c = np.argwhere(air_mask).mean(axis=0).astype(int).tolist()

    views = [
        (lambda v: v[c[0], :, :].T, "sagittal"),
        (lambda v: v[:, c[1], :].T, "coronal"),
        (lambda v: v[:, :, c[2]].T, "axial"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for col, (slicer, name) in enumerate(views):
        axes[col].imshow(slicer(intensity), cmap="gray", origin="lower", vmin=0, vmax=1)
        ov = slicer(air_mask.astype(float))
        axes[col].imshow(ov, cmap="autumn", origin="lower", alpha=0.55 * (ov > 0.1), vmin=0, vmax=1)
        axes[col].set_title(name)
        axes[col].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    audit_df = pd.read_csv(AUDIT_CSV)
    roi_df = pd.read_csv(ROI_BBOX_CSV)

    positives = audit_df[audit_df.label == "positive"]["filepath"].tolist()
    negatives = audit_df[audit_df.label == "negative"]["filepath"].tolist()

    print(f"Scanning {len(positives)} positive volumes for air-mask extent "
          f"(full body-mask + air-mask pipeline per volume -- may take a few minutes)...")
    records = []
    for i, fp in enumerate(positives):
        arr, body_mask, air_mask = compute_air_and_body(fp, audit_df, roi_df)
        records.append({
            "filepath": fp,
            "air_voxels": int(air_mask.sum()),
            "air_volume_ml": air_mask.sum() * (VOXEL_MM ** 3) / 1000.0,
            "centrality_mm": centrality_score(body_mask, air_mask),
        })
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(positives)} scanned")

    df = pd.DataFrame(records).sort_values("air_volume_ml", ascending=False)
    csv_path = os.path.join(OUT_DIR, "positive_air_mask_ranking.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved full ranking to {csv_path}")
    print(df.head(10).to_string(index=False))

    top_positive = df.head(N_TOP_POSITIVE)["filepath"].tolist()
    remaining_positive = df.iloc[N_TOP_POSITIVE:]["filepath"].tolist()
    rng = np.random.RandomState(5)
    random_positive = list(rng.choice(remaining_positive,
                                       size=min(N_RANDOM_POSITIVE, len(remaining_positive)), replace=False))
    random_negative = list(rng.choice(negatives, size=min(N_RANDOM_NEGATIVE, len(negatives)), replace=False))

    selected = (
        [(fp, "top_air_positive") for fp in top_positive] +
        [(fp, "random_positive") for fp in random_positive] +
        [(fp, "random_negative") for fp in random_negative]
    )

    print(f"\nRendering QC panels for {len(selected)} volumes "
          f"({N_TOP_POSITIVE} largest-air positives, {len(random_positive)} random positives, "
          f"{len(random_negative)} random negatives)...")
    for fp, category in selected:
        arr, body_mask, air_mask = compute_air_and_body(fp, audit_df, roi_df)
        row = df[df.filepath == fp]
        vol_ml = row["air_volume_ml"].iloc[0] if len(row) else air_mask.sum() * VOXEL_MM ** 3 / 1000.0
        centrality = row["centrality_mm"].iloc[0] if len(row) else centrality_score(body_mask, air_mask)
        filename = os.path.basename(fp)
        title = f"{filename} [{category}]  air_vol={vol_ml:.1f} mL  centrality={centrality:.1f} mm"
        out_path = os.path.join(OUT_DIR, f"{category}__{filename.replace('.nii.gz', '')}.png")
        render_panel(arr, air_mask, title, out_path)
        print(f"  [{category}] {filename}: air_vol={vol_ml:.1f}mL, centrality={centrality:.1f}mm -> {out_path}")

    print(f"\nAll panels saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
