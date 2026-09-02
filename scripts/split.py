"""
Step 8: patient-level leak-free train/val/test split.

The 9 multi-scan patients (3 with conflicting labels across scans) must stay
entirely within one split -- split at the patient level, then expand back to
individual scan files, never the reverse.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")


def patient_level_split(audit_csv: str = AUDIT_CSV, train_frac: float = 0.7,
                         val_frac: float = 0.15, seed: int = 42):
    """
    Returns a dict {'train': [...filepaths], 'val': [...], 'test': [...]}.
    Splitting is done on unique patient_id, using each patient's *first* scan
    label to stratify (multi-scan patients with conflicting labels are kept
    together regardless -- stratification is just for balance, not a
    correctness requirement).
    """
    df = pd.read_csv(audit_csv)
    rng = np.random.RandomState(seed)

    # one row per patient: use majority label (or first scan) for stratification
    patient_label = df.groupby("patient_id")["label"].agg(lambda s: s.mode().iloc[0])
    patients = patient_label.index.to_numpy()

    pos_patients = patient_label[patient_label == "positive"].index.to_numpy()
    neg_patients = patient_label[patient_label == "negative"].index.to_numpy()
    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    def split_group(arr):
        n = len(arr)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        return arr[:n_train], arr[n_train:n_train + n_val], arr[n_train + n_val:]

    pos_train, pos_val, pos_test = split_group(pos_patients)
    neg_train, neg_val, neg_test = split_group(neg_patients)

    train_patients = set(pos_train) | set(neg_train)
    val_patients = set(pos_val) | set(neg_val)
    test_patients = set(pos_test) | set(neg_test)

    assert not (train_patients & val_patients)
    assert not (train_patients & test_patients)
    assert not (val_patients & test_patients)
    assert train_patients | val_patients | test_patients == set(patients)

    splits = {}
    for name, pset in [("train", train_patients), ("val", val_patients), ("test", test_patients)]:
        rows = df[df["patient_id"].isin(pset)]
        splits[name] = rows["filepath"].tolist()

    return splits, {
        "train": train_patients, "val": val_patients, "test": test_patients,
    }


def _verify_multiscan_integrity(audit_csv: str = AUDIT_CSV, splits_patients=None):
    df = pd.read_csv(audit_csv)
    counts = df.groupby("patient_id").size()
    multi = counts[counts > 1].index.tolist()
    assert len(multi) == 9, f"expected 9 multi-scan patients, found {len(multi)}"

    for pid in multi:
        found_in = [name for name, pset in splits_patients.items() if pid in pset]
        assert len(found_in) == 1, f"patient {pid} spans {found_in}, must be exactly 1 split"
    return multi


if __name__ == "__main__":
    splits, splits_patients = patient_level_split()

    for name in ["train", "val", "test"]:
        files = splits[name]
        pids = splits_patients[name]
        df = pd.read_csv(AUDIT_CSV)
        sub = df[df["filepath"].isin(files)]
        pos = (sub["label"] == "positive").sum()
        neg = (sub["label"] == "negative").sum()
        print(f"{name}: {len(files)} volumes ({len(pids)} patients), "
              f"{pos} positive / {neg} negative ({100*pos/len(files):.1f}% positive)")

    multi = _verify_multiscan_integrity(splits_patients=splits_patients)
    print(f"\nAll {len(multi)} multi-scan patients confined to a single split: OK")
    print("Multi-scan patient IDs:", multi)
