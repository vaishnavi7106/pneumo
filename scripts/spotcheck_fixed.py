"""Spot-check the specific volumes that were misclassified pre-fix, now that
audit_results.csv and roi_bboxes.csv have been corrected."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PneumoDataset, ROI_BBOX_CSV  # noqa: E402
from visualize_roi_crops import plot_case  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
OUT_DIR = os.path.join(ROOT, "scripts", "roi_investigation_png")

TARGET_FILES = [
    "20251028273-1.nii.gz", "20251028126-1.nii.gz",
    "20251028170-1.nii.gz", "20251028301-1.nii.gz",
]

os.makedirs(OUT_DIR, exist_ok=True)
audit = pd.read_csv(AUDIT_CSV)
roi_df = pd.read_csv(ROI_BBOX_CSV)
rows = audit[audit["filename"].isin(TARGET_FILES)]

ds = PneumoDataset(rows["filepath"].tolist(), patch_size=(224, 224, 224))

for _, row in rows.iterrows():
    match = roi_df[roi_df["filepath"] == row["filepath"]]
    roi_row = match.iloc[0] if len(match) and match.iloc[0].get("status") == "ok" else None
    assert roi_row is not None, f"no ROI bbox found for {row['filename']}"
    out_path = os.path.join(OUT_DIR, f"FIXED__{row['filename'].replace('.nii.gz', '')}.png")
    plot_case(row, roi_row, ds, out_path)
    print(f"{row['filename']}: extent_z={row['extent_z_mm']:.0f}mm -> saved {out_path}")
