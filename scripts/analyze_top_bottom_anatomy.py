"""
Real anatomical comparison (not just length-matching) between:
  - abdomen-only volumes: their OWN natural top/bottom scan boundaries
    (never cropped by us, so this is just what the acquisition covers)
  - extended-FOV volumes: the top/bottom edges of the new fixed 280mm
    z-window crop

For both groups, runs VISTA3D segmentation and reports which organ labels
are actually present (by voxel count, not eyeballing) in the topmost and
bottommost slices, so we can tell whether the two groups' crops start/end
at comparable anatomical landmarks or not.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roi_localizer import ROI_ORGAN_IDS, segment_organ_bbox_mm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
ROI_CSV = os.path.join(ROOT, "scripts", "roi_bboxes.csv")

ORGAN_NAMES = {
    1: "liver", 2: "kidney", 3: "spleen", 4: "pancreas", 5: "right kidney",
    7: "IVC", 8: "right adrenal", 9: "left adrenal", 10: "gallbladder",
    11: "esophagus", 12: "stomach", 13: "duodenum", 14: "left kidney",
    15: "bladder", 17: "portal/splenic vein", 18: "rectum", 19: "small bowel", 62: "colon",
}

EDGE_N_SLICES = 5  # how many voxel slices (at 1.5mm each = 7.5mm) to inspect at each edge
MIN_VOXELS = 5  # ignore an organ label if it appears in fewer than this many voxels (noise)


def organs_in_slice_range(label_map, z_lo, z_hi):
    """label_map: [X,Y,Z] array. Returns dict organ_id -> voxel_count within z in [z_lo, z_hi)."""
    sub = label_map[:, :, z_lo:z_hi]
    ids, counts = np.unique(sub, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts) if i != 0 and c >= MIN_VOXELS}


def analyze_extended(row, roi_row):
    """Top/bottom organ composition within the fixed 280mm crop window."""
    r = segment_organ_bbox_mm(row["filepath"], return_label_map=True)
    if r["status"] != "ok":
        return None
    label_map = r["label_map"]
    affine = r["label_map_affine"]
    shape = label_map.shape

    # convert the crop's [bbox_min_z, bbox_max_z] (mm) into voxel indices in
    # THIS label_map's own grid
    inv = np.linalg.inv(affine)
    z_mm_lo, z_mm_hi = roi_row["bbox_min_z"], roi_row["bbox_max_z"]
    corners = np.array([[0, 0, z_mm_lo, 1], [0, 0, z_mm_hi, 1]])
    vox = (inv @ corners.T).T[:, 2]
    z_vox_lo, z_vox_hi = int(np.floor(min(vox))), int(np.ceil(max(vox)))
    z_vox_lo = max(0, z_vox_lo)
    z_vox_hi = min(shape[2], z_vox_hi)

    if z_vox_hi - z_vox_lo < EDGE_N_SLICES * 2:
        return None

    bottom = organs_in_slice_range(label_map, z_vox_lo, z_vox_lo + EDGE_N_SLICES)
    top = organs_in_slice_range(label_map, z_vox_hi - EDGE_N_SLICES, z_vox_hi)
    return {"top": top, "bottom": bottom}


def analyze_abdomen_only(row):
    """Top/bottom organ composition at the volume's OWN natural (uncropped) boundaries."""
    r = segment_organ_bbox_mm(row["filepath"], return_label_map=True)
    if r["status"] != "ok":
        return None
    label_map = r["label_map"]
    shape = label_map.shape
    if shape[2] < EDGE_N_SLICES * 2:
        return None
    bottom = organs_in_slice_range(label_map, 0, EDGE_N_SLICES)
    top = organs_in_slice_range(label_map, shape[2] - EDGE_N_SLICES, shape[2])
    return {"top": top, "bottom": bottom}


def summarize(results, group_name):
    print(f"\n=== {group_name}: n={len(results)} ===")
    for edge in ["top", "bottom"]:
        organ_volume_count = {}  # organ_id -> number of volumes where it appears
        for res in results:
            for oid in res[edge]:
                organ_volume_count[oid] = organ_volume_count.get(oid, 0) + 1
        print(f"\n  {edge} edge -- organs present, by how many of {len(results)} volumes:")
        for oid, cnt in sorted(organ_volume_count.items(), key=lambda x: -x[1]):
            name = ORGAN_NAMES.get(oid, f"id{oid}")
            print(f"    {name:25s} (id={oid:3d}): {cnt}/{len(results)} volumes ({100*cnt/len(results):.0f}%)")
    return


def main(n_per_group: int = 12, seed: int = 1):
    audit = pd.read_csv(AUDIT_CSV)
    roi = pd.read_csv(ROI_CSV)

    extended = audit[audit.extent_z_mm >= 375].sample(n_per_group, random_state=seed)
    abdomen_only = audit[audit.extent_z_mm < 375].sample(min(n_per_group, (audit.extent_z_mm < 375).sum()),
                                                          random_state=seed)

    print(f"Sampling {len(extended)} extended-FOV and {len(abdomen_only)} abdomen-only volumes")

    ext_results = []
    for i, (_, row) in enumerate(extended.iterrows()):
        roi_row = roi[roi.filepath == row["filepath"]]
        if not len(roi_row) or roi_row.iloc[0]["status"] != "ok":
            continue
        print(f"[extended {i+1}/{len(extended)}] {row['filename']} ...")
        res = analyze_extended(row, roi_row.iloc[0])
        if res:
            res["filename"] = row["filename"]
            ext_results.append(res)

    ab_results = []
    for i, (_, row) in enumerate(abdomen_only.iterrows()):
        print(f"[abdomen-only {i+1}/{len(abdomen_only)}] {row['filename']} ...")
        res = analyze_abdomen_only(row)
        if res:
            res["filename"] = row["filename"]
            ab_results.append(res)

    summarize(ext_results, "EXTENDED-FOV (fixed 280mm crop window)")
    summarize(ab_results, "ABDOMEN-ONLY (natural, uncropped scan boundaries)")

    # save raw per-volume detail too
    import json
    with open("scripts/top_bottom_anatomy.json", "w") as f:
        json.dump({"extended": ext_results, "abdomen_only": ab_results}, f, indent=2, default=str)
    print("\nSaved raw per-volume detail to scripts/top_bottom_anatomy.json")


if __name__ == "__main__":
    main()
