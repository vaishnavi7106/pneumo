"""
Bbox consistency analysis across all 333 extended-FOV volumes:
1. Distribution of bbox x/y/z physical extent (mm), flag outliers vs median.
2. Compare cropped-group bbox z-coverage against the natural (uncropped)
   z-extent of the abdomen-only group -- do they land in a similar range?
3. Report any segmentation failures / suspicious outlier bboxes explicitly.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

audit = pd.read_csv("scripts/audit_results.csv")
roi = pd.read_csv("scripts/roi_bboxes.csv")

roi["bbox_x_extent"] = roi["bbox_max_x"] - roi["bbox_min_x"]
roi["bbox_y_extent"] = roi["bbox_max_y"] - roi["bbox_min_y"]
roi["bbox_z_extent"] = roi["bbox_max_z"] - roi["bbox_min_z"]

print(f"Total extended-FOV volumes with ROI bbox: {len(roi)}")
print(f"status counts:\n{roi['status'].value_counts()}\n")

failures = roi[roi["status"] != "ok"]
print(f"Segmentation failures (status != 'ok'): {len(failures)}")
if len(failures):
    print(failures[["filename", "status"]].to_string())

low_organ_count = roi[roi["n_found_organs"] < 14]
print(f"\nVolumes with fewer than 14/18 organs found (weak segmentation signal): {len(low_organ_count)}")
if len(low_organ_count):
    print(low_organ_count[["filename", "n_found_organs"]].to_string())

print("\n=== bbox extent distributions (mm) ===")
for dim in ["x", "y", "z"]:
    col = f"bbox_{dim}_extent"
    print(f"\n{dim}-extent: median={roi[col].median():.1f}, mean={roi[col].mean():.1f}, "
          f"std={roi[col].std():.1f}, min={roi[col].min():.1f}, max={roi[col].max():.1f}")

# flag outliers: > 2.5 std from median (robust-ish threshold) in any dimension
outlier_rows = []
for dim in ["x", "y", "z"]:
    col = f"bbox_{dim}_extent"
    if roi[col].std() < 1e-6:  # effectively constant (allow for float rounding noise)
        print(f"\n{dim}-extent outliers: skipped (constant value {roi[col].iloc[0]:.1f}mm across all volumes -- "
              f"expected for a fixed-size window, not meaningful to flag outliers)")
        outlier_rows.append(pd.Series([], dtype=str))
        continue
    med = roi[col].median()
    mad = (roi[col] - med).abs().median() * 1.4826  # robust std estimate
    if mad == 0:
        mad = roi[col].std()
    lo, hi = med - 3 * mad, med + 3 * mad
    outliers = roi[(roi[col] < lo) | (roi[col] > hi)]
    print(f"\n{dim}-extent outliers (outside [{lo:.1f}, {hi:.1f}]mm, robust 3-MAD): {len(outliers)}")
    if len(outliers):
        print(outliers[["filename", col, "n_found_organs"]].sort_values(col).to_string())
    outlier_rows.append(outliers["filename"])

all_outlier_files = pd.concat(outlier_rows).unique()
print(f"\nTotal unique volumes flagged as bbox-size outliers (any dimension): {len(all_outlier_files)}")

# compare cropped bbox z-coverage vs abdomen-only natural z-extent
abdomen_only = audit[audit["extent_z_mm"] < 375]
print(f"\n=== z-extent comparison: cropped bbox vs abdomen-only natural extent ===")
print(f"Abdomen-only (uncropped) natural extent_z_mm: "
      f"median={abdomen_only['extent_z_mm'].median():.1f}, "
      f"mean={abdomen_only['extent_z_mm'].mean():.1f}, "
      f"range=[{abdomen_only['extent_z_mm'].min():.1f}, {abdomen_only['extent_z_mm'].max():.1f}]")
print(f"Extended-FOV (cropped) bbox z_extent: "
      f"median={roi['bbox_z_extent'].median():.1f}, "
      f"mean={roi['bbox_z_extent'].mean():.1f}, "
      f"range=[{roi['bbox_z_extent'].min():.1f}, {roi['bbox_z_extent'].max():.1f}]")

# plot distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, dim in zip(axes.flat[:3], ["x", "y", "z"]):
    col = f"bbox_{dim}_extent"
    n_bins = 40 if roi[col].std() > 1e-6 else 1
    ax.hist(roi[col], bins=n_bins, color="steelblue", edgecolor="black")
    ax.axvline(roi[col].median(), color="red", linestyle="--", label=f"median={roi[col].median():.0f}mm")
    ax.set_title(f"bbox {dim}-extent (mm), n={len(roi)}")
    ax.set_xlabel("mm")
    ax.legend()

ax = axes.flat[3]
ax.hist(abdomen_only["extent_z_mm"], bins=20, alpha=0.6, label="abdomen-only (uncropped, natural)",
         color="orange", density=True)
if roi["bbox_z_extent"].std() > 1e-6:
    ax.hist(roi["bbox_z_extent"], bins=40, alpha=0.6, label="extended-FOV (cropped bbox)",
             color="steelblue", density=True)
else:
    # fixed-size window -> a single value, not a distribution; draw as a line instead
    z_val = roi["bbox_z_extent"].mean()
    ax.axvline(z_val, color="steelblue", linewidth=3,
               label=f"extended-FOV (fixed window={z_val:.0f}mm, all 333 volumes)")
ax.set_title("z-extent: cropped bbox vs abdomen-only natural extent")
ax.set_xlabel("mm")
ax.legend()

fig.tight_layout()
fig.savefig("scripts/bbox_consistency.png", dpi=120)
print("\nSaved plot to scripts/bbox_consistency.png")
