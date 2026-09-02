"""
Compute ROI bboxes for the 75 volumes that were misclassified as abdomen-only
(and thus never sent through VISTA3D segmentation) due to the axis-mapping
bug in the old audit_dataset.py -- now fixed. Appends to the existing
roi_bboxes.csv rather than recomputing the 258 already-correct entries.
"""
import csv
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roi_localizer import ROI_ORGAN_IDS, segment_organ_bbox_mm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
OUT_CSV = os.path.join(ROOT, "scripts", "roi_bboxes.csv")

FIELDNAMES = ["filepath", "filename", "status", "n_found_organs", "found_labels", "elapsed_s",
              "bbox_min_x", "bbox_min_y", "bbox_min_z", "bbox_max_x", "bbox_max_y", "bbox_max_z", "error"]


def main():
    audit = pd.read_csv(AUDIT_CSV)
    roi = pd.read_csv(OUT_CSV)

    extended_now = set(audit[audit.extent_z_mm >= 375]["filepath"])
    already_done = set(roi[roi.status == "ok"]["filepath"])
    missing = sorted(extended_now - already_done)

    print(f"{len(missing)} volumes need new ROI bbox computation")
    missing_rows = audit[audit["filepath"].isin(missing)]

    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        pbar = tqdm(missing_rows.itertuples(index=False), total=len(missing_rows),
                    desc="ROI bbox (missing 75)", file=sys.stdout)
        for row in pbar:
            t0 = time.time()
            try:
                r = segment_organ_bbox_mm(row.filepath)
            except Exception as e:  # noqa: BLE001
                pbar.write(f"{row.filename}: ERROR {type(e).__name__}: {e}")
                rec = {"filepath": row.filepath, "filename": row.filename, "status": "error", "error": repr(e)}
                writer.writerow(rec)
                f.flush()
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
            pbar.set_postfix(status=r["status"], organs=len(r["found_labels"]), s=f"{elapsed:.1f}")

    print(f"\nDone. Appended to {OUT_CSV}")
    final = pd.read_csv(OUT_CSV)
    print(final["status"].value_counts())
    print(f"Total rows: {len(final)} (should be 333)")


if __name__ == "__main__":
    main()
