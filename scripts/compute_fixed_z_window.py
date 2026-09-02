"""
Replace the union-bbox z-cropping with a fixed-size physical z-window
centered on the core-organ centroid, biased upward (toward the diaphragm/
liver dome) since subphrenic free air -- the classic pneumoperitoneum
presentation -- collects at the top of the abdominal cavity and must not be
cropped out. x/y cropping is left unchanged (union bbox), since it does not
show the same disjoint-distribution problem z did.

Reruns VISTA3D segmentation for all 333 extended-FOV volumes (needed because
the per-volume core centroid was never previously computed/cached) and
overwrites roi_bboxes.csv's z bounds in place, keeping x/y bounds as-is.
"""
import csv
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roi_localizer import segment_organ_bbox_mm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
ROI_CSV = os.path.join(ROOT, "scripts", "roi_bboxes.csv")
OUT_CSV = os.path.join(ROOT, "scripts", "roi_bboxes.csv")  # overwrite in place

# Fixed z-window: 280mm total, biased upward (superior) from the core
# centroid -- chosen from the abdomen-only group's natural range (~250-320mm,
# see bbox_consistency analysis) with an asymmetric split so the window
# reaches further toward the diaphragm/liver dome than toward the pelvis.
SUPERIOR_MARGIN_MM = 180.0  # toward diaphragm/liver dome (increasing S)
INFERIOR_MARGIN_MM = 100.0  # toward pelvis
WINDOW_SIZE_MM = SUPERIOR_MARGIN_MM + INFERIOR_MARGIN_MM  # 280mm

FIELDNAMES = ["filepath", "filename", "status", "n_found_organs", "found_labels", "elapsed_s",
              "bbox_min_x", "bbox_min_y", "bbox_min_z", "bbox_max_x", "bbox_max_y", "bbox_max_z",
              "core_centroid_z_mm", "z_window_method", "error"]


def main():
    audit = pd.read_csv(AUDIT_CSV)
    extended = audit[audit.extent_z_mm >= 375].reset_index(drop=True)
    print(f"{len(extended)} extended-FOV volumes to reprocess with fixed z-window "
          f"(superior_margin={SUPERIOR_MARGIN_MM}mm, inferior_margin={INFERIOR_MARGIN_MM}mm, "
          f"total={WINDOW_SIZE_MM}mm)")

    records = []
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        pbar = tqdm(extended.itertuples(index=False), total=len(extended),
                    desc="fixed z-window", file=sys.stdout)
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
                # x/y unchanged: from the existing all-organ union bbox
                rec["bbox_min_x"], rec["bbox_min_y"], _ = r["bbox_min_mm"]
                rec["bbox_max_x"], rec["bbox_max_y"], _ = r["bbox_max_mm"]

                core_centroid = r.get("core_centroid_mm")
                if core_centroid is not None:
                    centroid_z = core_centroid[2]
                    rec["bbox_min_z"] = centroid_z - INFERIOR_MARGIN_MM
                    rec["bbox_max_z"] = centroid_z + SUPERIOR_MARGIN_MM
                    rec["core_centroid_z_mm"] = centroid_z
                    rec["z_window_method"] = "fixed_window_core_centroid"
                else:
                    # no core organs found (shouldn't happen given prior 333/333
                    # success) -- fall back to the old union-bbox z as a safety net
                    rec["bbox_min_z"] = r["bbox_min_mm"][2]
                    rec["bbox_max_z"] = r["bbox_max_mm"][2]
                    rec["z_window_method"] = "fallback_union_bbox_no_core_organs"
                    pbar.write(f"{row.filename}: WARNING no core organs found, using union-bbox z fallback")

            writer.writerow(rec)
            f.flush()
            records.append(rec)
            pbar.set_postfix(status=r["status"], organs=len(r["found_labels"]))

    out_df = pd.DataFrame(records)
    print(f"\nSaved fixed-z-window ROI bboxes to {OUT_CSV}")
    print(out_df["status"].value_counts())
    if "z_window_method" in out_df:
        print(out_df["z_window_method"].value_counts())


if __name__ == "__main__":
    main()
