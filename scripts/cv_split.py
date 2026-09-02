"""
Patient-level 3-fold cross-validation split using StratifiedGroupKFold
(group=patient_id, so multi-scan patients -- including the 9 with 2 scans --
stay entirely within one fold). Each fold's non-test patients are further
split into an internal train/val (85/15, also patient-grouped + stratified)
for early stopping and threshold tuning, so the fold's held-out test set is
never touched until final evaluation.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")


def make_cv_folds(n_splits: int = 3, seed: int = 42, val_frac: float = 0.15,
                   audit_csv: str = AUDIT_CSV):
    df = pd.read_csv(audit_csv)
    patient_label = df.groupby("patient_id")["label"].agg(lambda s: s.mode().iloc[0])
    patients = patient_label.index.to_numpy()
    labels = (patient_label == "positive").astype(int).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # StratifiedGroupKFold needs X, y, groups -- groups IS patients here since each
    # patient contributes exactly one row to this stratification frame
    fold_assignment = np.zeros(len(patients), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(sgkf.split(patients, labels, groups=patients)):
        fold_assignment[test_idx] = fold_idx

    folds = []
    rng = np.random.RandomState(seed)
    for fold_idx in range(n_splits):
        test_patients = set(patients[fold_assignment == fold_idx])
        train_val_patients = set(patients[fold_assignment != fold_idx])

        # further split train_val_patients into train/val, stratified by label, patient-grouped
        tv_patients = np.array(sorted(train_val_patients))
        tv_labels = patient_label.loc[tv_patients].to_numpy()

        pos_p = tv_patients[tv_labels == "positive"]
        neg_p = tv_patients[tv_labels == "negative"]
        rng.shuffle(pos_p)
        rng.shuffle(neg_p)

        def split_val(arr):
            n_val = max(1, int(round(len(arr) * val_frac)))
            return arr[n_val:], arr[:n_val]  # train, val

        pos_train, pos_val = split_val(pos_p)
        neg_train, neg_val = split_val(neg_p)
        train_patients = set(pos_train) | set(neg_train)
        val_patients = set(pos_val) | set(neg_val)

        assert not (train_patients & val_patients)
        assert not (train_patients & test_patients)
        assert not (val_patients & test_patients)
        assert train_patients | val_patients | test_patients == set(patients)

        splits = {}
        for name, pset in [("train", train_patients), ("val", val_patients), ("test", test_patients)]:
            rows = df[df["patient_id"].isin(pset)]
            splits[name] = rows["filepath"].tolist()

        folds.append({
            "fold": fold_idx, "splits": splits,
            "patients": {"train": train_patients, "val": val_patients, "test": test_patients},
        })

    return folds


def _verify_multiscan_integrity(folds, audit_csv: str = AUDIT_CSV):
    """A multi-scan patient must never have their scans split across two
    DIFFERENT splits within the SAME fold (e.g. scan1 in fold0-train, scan2
    in fold0-val). Appearing in fold0-train AND fold1-test is correct k-fold
    behavior (every non-test-fold patient contributes to other folds'
    train/val) -- only a same-fold split would be a real leakage bug."""
    df = pd.read_csv(audit_csv)
    counts = df.groupby("patient_id").size()
    multi = counts[counts > 1].index.tolist()
    assert len(multi) == 9, f"expected 9 multi-scan patients, found {len(multi)}"

    for pid in multi:
        for fold in folds:
            splits_containing_pid = [name for name, pset in fold["patients"].items() if pid in pset]
            assert len(splits_containing_pid) <= 1, (
                f"patient {pid} split across {splits_containing_pid} WITHIN fold {fold['fold']} -- leakage bug"
            )

    # also confirm each multi-scan patient's test-fold assignment is unique (already
    # checked in aggregate above, but re-verify per-patient for a precise error message)
    for pid in multi:
        test_folds = [fold["fold"] for fold in folds if pid in fold["patients"]["test"]]
        assert len(test_folds) == 1, f"patient {pid} is a test patient in folds {test_folds}, must be exactly 1"

    return multi


if __name__ == "__main__":
    folds = make_cv_folds()
    df = pd.read_csv(AUDIT_CSV)

    for fold in folds:
        print(f"\n=== FOLD {fold['fold']} ===")
        for name in ["train", "val", "test"]:
            files = fold["splits"][name]
            sub = df[df["filepath"].isin(files)]
            pos = (sub["label"] == "positive").sum()
            neg = (sub["label"] == "negative").sum()
            print(f"  {name}: {len(files)} volumes ({len(fold['patients'][name])} patients), "
                  f"{pos} positive / {neg} negative ({100*pos/len(files):.1f}% positive)")

    # cross-fold sanity: every patient assigned to exactly one fold's test set
    all_test_patients = set()
    for fold in folds:
        overlap = all_test_patients & fold["patients"]["test"]
        assert not overlap, f"patient overlap across fold test sets: {overlap}"
        all_test_patients |= fold["patients"]["test"]
    df_all_patients = set(df["patient_id"].unique())
    assert all_test_patients == df_all_patients, "not every patient assigned to exactly one fold's test set"
    print(f"\nAll {len(all_test_patients)} patients each assigned to exactly one fold's test set: OK")

    multi = _verify_multiscan_integrity(folds)
    print(f"All {len(multi)} multi-scan patients confined to a single (fold, split): OK")
    print("Multi-scan patient IDs:", multi)
