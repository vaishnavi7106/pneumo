"""
Step 4: cross-check the 363-volume dataset against the ground-truth xlsx report.
Match filename {patientID}-{scanIdx}.nii.gz to report rows sorted by scan date
per patient (NOT a naive patient_id-only merge -- that caused duplicate-row
bugs previously).
"""
import io
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(ROOT, "2025783_Analysis_Result_clean_gas_result_with_free_air_location.xlsx")
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
OUT_LOG = os.path.join(ROOT, "scripts", "crosscheck_report.txt")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PATIENT_COL = "匿名病歷號"
LABEL_COL = "Pneumoperitoneum_Label"
KEEP_COL = "Keep_In_Dataset"
DATE_COL = "電腦斷層掃描日期"


def main():
    log = []

    def p(msg=""):
        print(msg)
        log.append(str(msg))

    gt = pd.read_excel(XLSX)
    p(f"Ground-truth report shape: {gt.shape}")
    p(f"Columns: {list(gt.columns)}")

    kept = gt[gt[KEEP_COL] == True]  # noqa: E712
    p(f"\nRows with Keep_In_Dataset==True: {len(kept)}")

    audit = pd.read_csv(AUDIT_CSV)
    p(f"Volumes on disk: {len(audit)}")

    # sort GT rows per patient by scan date, assign chronological scan_idx
    kept = kept.copy()
    kept[DATE_COL] = pd.to_datetime(kept[DATE_COL], errors="coerce")
    kept = kept.sort_values([PATIENT_COL, DATE_COL])
    kept["scan_idx_gt"] = kept.groupby(PATIENT_COL).cumcount() + 1

    kept["patient_id_str"] = kept[PATIENT_COL].astype(str)
    audit["patient_id_str"] = audit["patient_id"].astype(str)
    audit["scan_idx_int"] = audit["scan_idx"].astype(int)

    merged = audit.merge(
        kept,
        left_on=["patient_id_str", "scan_idx_int"],
        right_on=["patient_id_str", "scan_idx_gt"],
        how="left",
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"]
    p(f"\nUnmatched volumes (no GT row found): {len(unmatched)}")
    if len(unmatched):
        p(unmatched[["filename", "patient_id_str", "scan_idx_int"]].to_string())

    # duplicate check: each on-disk file should match exactly one GT row
    dup_check = merged.groupby("filename").size()
    dups = dup_check[dup_check > 1]
    p(f"\nFiles matched to >1 GT row (duplicate-merge bug check): {len(dups)}")
    if len(dups):
        p(dups.to_string())

    # label mismatch: on-disk folder label vs GT label
    def gt_label_to_str(v):
        if pd.isna(v):
            return None
        return "positive" if str(v).strip().lower() in ("yes", "y", "1", "true") else "negative"

    merged["gt_label"] = merged[LABEL_COL].apply(gt_label_to_str)
    mismatches = merged[
        merged["gt_label"].notna() & (merged["gt_label"] != merged["label"])
    ]
    p(f"\nLabel mismatches (on-disk folder vs GT report): {len(mismatches)}")
    if len(mismatches):
        p(mismatches[["filename", "label", "gt_label"]].to_string())

    # coverage: every GT-kept row should have a matching file
    kept["patient_scan_key"] = kept["patient_id_str"] + "-" + kept["scan_idx_gt"].astype(str)
    audit["patient_scan_key"] = audit["patient_id_str"] + "-" + audit["scan_idx_int"].astype(str)
    missing_files = kept[~kept["patient_scan_key"].isin(set(audit["patient_scan_key"]))]
    p(f"\nGT-kept rows with no matching file on disk: {len(missing_files)}")
    if len(missing_files):
        p(missing_files[[PATIENT_COL, "scan_idx_gt", LABEL_COL]].to_string())

    p(f"\nFinal matched count: {(merged['_merge']=='both').sum()} / {len(audit)}")

    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    p(f"\nSaved report to {OUT_LOG}")


if __name__ == "__main__":
    main()
