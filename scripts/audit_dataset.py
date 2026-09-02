"""
Dataset audit for pneumoperitoneum CT classification project.
Re-runs geometry/spacing/orientation/HU/quality checks against merged/Pneumo_Positive
and merged/Pneumo_Negative, per PROJECT_BRIEFING.md step 3.
"""
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS_DIR = os.path.join(ROOT, "merged", "Pneumo_Positive")
NEG_DIR = os.path.join(ROOT, "merged", "Pneumo_Negative")
OUT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")


def list_files():
    pos = sorted(glob.glob(os.path.join(POS_DIR, "*.nii.gz")))
    neg = sorted(glob.glob(os.path.join(NEG_DIR, "*.nii.gz")))
    return [(p, "positive") for p in pos] + [(p, "negative") for p in neg]


def parse_filename(path):
    base = os.path.basename(path)
    stem = base[:-len(".nii.gz")] if base.endswith(".nii.gz") else base
    if "-" in stem:
        patient_id, scan_idx = stem.rsplit("-", 1)
    else:
        patient_id, scan_idx = stem, "1"
    return patient_id, scan_idx


def audit_one(path, label):
    row = {
        "filepath": path,
        "filename": os.path.basename(path),
        "label": label,
    }
    patient_id, scan_idx = parse_filename(path)
    row["patient_id"] = patient_id
    row["scan_idx"] = scan_idx

    try:
        row["file_size_mb"] = os.path.getsize(path) / (1024 * 1024)
        img = nib.load(path)
        hdr = img.header
        shape = img.shape
        zooms = hdr.get_zooms()[:3]
        affine = img.affine
        axcodes = nib.aff2axcodes(affine)
        orientation = "".join(axcodes)

        # Orientation-aware axis mapping: array axis index 2 is NOT always the
        # slice (S-I) axis -- e.g. for LIP-oriented volumes it's axis 1. Map
        # each array axis to its anatomical group (x=L/R, y=A/P, z=S/I) via
        # the actual affine-derived orientation codes instead of assuming a
        # fixed array index, so shape_z/spacing_z/extent_z_mm are always the
        # true superior-inferior (slice) values regardless of on-disk layout.
        axis_group = {"L": "x", "R": "x", "A": "y", "P": "y", "S": "z", "I": "z"}
        group_to_axis = {axis_group[code]: i for i, code in enumerate(axcodes)}

        for g in ("x", "y", "z"):
            axis_i = group_to_axis[g]
            row[f"shape_{g}"] = shape[axis_i]
            row[f"spacing_{g}"] = zooms[axis_i]

        row["orientation"] = orientation
        row["extent_z_mm"] = (
            row["shape_z"] * row["spacing_z"] if row["shape_z"] and row["spacing_z"] else None
        )
        row["on_disk_dtype"] = str(img.get_data_dtype())

        data = img.get_fdata(dtype=np.float32)
        row["min_hu"] = float(np.min(data))
        row["max_hu"] = float(np.max(data))
        row["mean_hu"] = float(np.mean(data))
        row["has_nan"] = bool(np.isnan(data).any())
        row["has_inf"] = bool(np.isinf(data).any())
        row["is_constant"] = bool(np.ptp(data) == 0)
        row["load_error"] = ""
    except Exception as e:  # noqa: BLE001
        row["load_error"] = repr(e)

    return row


def main():
    files = list_files()
    print(f"Found {len(files)} volumes ({sum(1 for _, l in files if l=='positive')} positive, "
          f"{sum(1 for _, l in files if l=='negative')} negative)")

    rows = [audit_one(p, l) for p, l in tqdm(files)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved audit to {OUT_CSV}")

    errors = df[df["load_error"] != ""]
    print(f"\nLoad errors: {len(errors)}")
    if len(errors):
        print(errors[["filename", "load_error"]])

    print(f"\nlabel counts:\n{df['label'].value_counts()}")

    multi = df.groupby("patient_id").size()
    multi_patients = multi[multi > 1]
    print(f"\nMulti-scan patients: {len(multi_patients)}")

    print("\nOrientation codes:")
    print(df["orientation"].value_counts())

    print("\ndtype counts:")
    print(df["on_disk_dtype"].value_counts())

    print("\nHU range summary:")
    print(df[["min_hu", "max_hu"]].describe())

    print("\nextent_z (mm) summary:")
    print(df["extent_z_mm"].describe())
    abdomen_only = (df["extent_z_mm"] >= 150) & (df["extent_z_mm"] <= 370)
    extended = (df["extent_z_mm"] >= 385) & (df["extent_z_mm"] <= 685)
    print(f"abdomen-only (150-370mm): {abdomen_only.sum()}")
    print(f"extended-FOV (385-685mm): {extended.sum()}")
    gap = (df["extent_z_mm"] > 370) & (df["extent_z_mm"] < 385)
    print(f"in gap (370-385mm): {gap.sum()}")

    print("\nThick-slice volumes (spacing_z > 10mm):")
    thick = df[df["spacing_z"] > 10]
    print(thick[["filename", "label", "spacing_z"]])

    print("\nQuality flags:")
    print(f"NaN: {df['has_nan'].sum()}, Inf: {df['has_inf'].sum()}, constant: {df['is_constant'].sum()}")


if __name__ == "__main__":
    main()
