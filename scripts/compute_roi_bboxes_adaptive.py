"""
Batch-compute ROI bboxes for all extended-FOV volumes using the new adaptive
crop (adaptive_roi_crop.py) instead of the VISTA3D-segmentation-based fixed
z-window (roi_localizer.py / compute_fixed_z_window.py).

Writes the SAME schema dataset.py already reads (bbox_min/max_x/y/z in
physical RAS mm, status=='ok' gate) to a SEPARATE file
(roi_bboxes_adaptive.csv) -- does not touch roi_bboxes.csv or dataset.py.
Swap it in later (rename, or point ROI_BBOX_CSV at this file) only after the
downstream shortcut-learning / sanity checks pass.

No VISTA3D inference needed here -- pure HU-geometry, runs on CPU, should be
much faster per volume than the segmentation-based approach.
"""
import csv
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_roi_crop import bbox_vox_to_mm, compute_tight_3axis_bbox, load_ras_native  # noqa: E402
from dataset import EXTENDED_FOV_THRESHOLD_MM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
OUT_CSV = os.path.join(ROOT, "scripts", "roi_bboxes_adaptive.csv")

FIELDNAMES = ["filepath", "filename", "status", "elapsed_s", "volume_reduction",
              "bbox_min_x", "bbox_min_y", "bbox_min_z", "bbox_max_x", "bbox_max_y", "bbox_max_z", "error"]


def main():
    audit = pd.read_csv(AUDIT_CSV)
    extended = audit[audit.extent_z_mm >= EXTENDED_FOV_THRESHOLD_MM].reset_index(drop=True)
    print(f"{len(extended)} extended-FOV volumes to process with the adaptive 3-axis crop")

    records = []
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        pbar = tqdm(extended.itertuples(index=False), total=len(extended),
                    desc="adaptive ROI crop", file=sys.stdout)
        for row in pbar:
            t0 = time.time()
            try:
                hu_array, affine, zooms = load_ras_native(row.filepath)
                bbox, margin_bbox, mask, diag = compute_tight_3axis_bbox(hu_array, zooms)
                bbox_min_mm, bbox_max_mm = bbox_vox_to_mm(affine, bbox)
                vol_reduction = 1.0 - (int(__import__("numpy").prod(
                    [s.stop - s.start for s in bbox])) / hu_array.size)
            except Exception as e:  # noqa: BLE001
                pbar.write(f"{row.filename}: ERROR {type(e).__name__}: {e}")
                rec = {"filepath": row.filepath, "filename": row.filename, "status": "error", "error": repr(e)}
                writer.writerow(rec)
                f.flush()
                records.append(rec)
                continue
            elapsed = time.time() - t0

            rec = {
                "filepath": row.filepath, "filename": row.filename, "status": "ok",
                "elapsed_s": elapsed, "volume_reduction": vol_reduction,
                "bbox_min_x": bbox_min_mm[0], "bbox_min_y": bbox_min_mm[1], "bbox_min_z": bbox_min_mm[2],
                "bbox_max_x": bbox_max_mm[0], "bbox_max_y": bbox_max_mm[1], "bbox_max_z": bbox_max_mm[2],
            }
            writer.writerow(rec)
            f.flush()
            records.append(rec)
            pbar.set_postfix(reduction=f"{vol_reduction:.1%}")

    out_df = pd.DataFrame(records)
    print(f"\nSaved adaptive ROI bboxes to {OUT_CSV}")
    print(out_df["status"].value_counts())
    ok = out_df[out_df["status"] == "ok"]
    if len(ok):
        print(f"Mean volume reduction: {ok['volume_reduction'].mean():.1%}")
        print(f"Min volume reduction: {ok['volume_reduction'].min():.1%}")
        print(f"Mean elapsed: {ok['elapsed_s'].mean():.2f}s/volume")


if __name__ == "__main__":
    main()
