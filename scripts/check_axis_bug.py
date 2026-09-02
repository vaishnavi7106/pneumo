"""
Investigate: does audit_dataset.py's naive `shape_z = shape[2]` /
`spacing_z = zooms[2]` correctly identify the true superior-inferior (slice)
axis for every orientation code, or does it silently use the wrong axis for
some orientations (breaking extent_z_mm and the abdomen-only/extended-FOV
classification that ROI cropping depends on)?
"""
import nibabel as nib
import numpy as np
import pandas as pd

df = pd.read_csv("scripts/audit_results.csv")

records = []
for _, row in df.iterrows():
    img = nib.load(row["filepath"])
    axcodes = nib.aff2axcodes(img.affine)
    # find which array axis corresponds to the S-I (Superior/Inferior) direction
    si_axis = None
    for i, code in enumerate(axcodes):
        if code in ("S", "I"):
            si_axis = i
            break
    naive_axis = 2  # what audit_dataset.py assumed

    zooms = img.header.get_zooms()[:3]
    shape = img.shape[:3]
    true_shape_z = shape[si_axis]
    true_spacing_z = zooms[si_axis]
    true_extent_z = true_shape_z * true_spacing_z

    mismatch = si_axis != naive_axis
    records.append({
        "filename": row["filename"], "orientation": row["orientation"],
        "si_axis_true": si_axis, "axis_mismatch": mismatch,
        "recorded_shape_z": row["shape_z"], "true_shape_z": true_shape_z,
        "recorded_extent_z_mm": row["extent_z_mm"], "true_extent_z_mm": true_extent_z,
        "recorded_fov": "extended" if row["extent_z_mm"] >= 375 else "abdomen-only",
        "true_fov": "extended" if true_extent_z >= 375 else "abdomen-only",
    })

out = pd.DataFrame(records)
out.to_csv("scripts/axis_bug_check.csv", index=False)

print(f"Total volumes: {len(out)}")
print(f"Volumes where naive axis-2 assumption is WRONG (si_axis != 2): {out['axis_mismatch'].sum()}")
print()
print("By orientation code:")
print(out.groupby("orientation")["axis_mismatch"].agg(["sum", "count"]))
print()
reclassified = out[out["recorded_fov"] != out["true_fov"]]
print(f"Volumes whose FOV classification CHANGES with correct axis: {len(reclassified)}")
if len(reclassified):
    print(reclassified[["filename", "orientation", "recorded_extent_z_mm", "true_extent_z_mm",
                         "recorded_fov", "true_fov"]].to_string())
