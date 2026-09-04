"""
Confirm augmentation only ever applies to the train split. Two checks:

1. Static wiring: make_dataloaders(..., augment_train=True) must produce a
   PneumoDataset with .augment=True for "train" and .augment=False for every
   other split name ("val", "test").
2. Behavioral: fetching the SAME index from a val/test dataset twice must
   return bit-identical tensors (no stochastic transform touched it); fetching
   the same index from the train dataset twice should usually differ (since
   augment_volume re-rolls flip/rotation/intensity every call).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import make_dataloaders  # noqa: E402
from split import patient_level_split  # noqa: E402


def main():
    splits, _ = patient_level_split()
    small_splits = {k: v[:3] for k, v in splits.items()}

    loaders = make_dataloaders(small_splits, batch_size=1, patch_size=(64, 64, 64), augment_train=True)

    for name, loader in loaders.items():
        ds = loader.dataset
        expected = (name == "train")
        assert ds.augment == expected, f"{name}: ds.augment={ds.augment}, expected {expected}"
        print(f"[wiring] {name}: ds.augment={ds.augment} (expected {expected}) -- OK")

    for name, loader in loaders.items():
        ds = loader.dataset
        x1, _, _ = ds[0]
        x2, _, _ = ds[0]
        identical = torch.equal(x1, x2)
        if name == "train":
            print(f"[behavior] train: repeated fetch identical={identical} "
                  f"(expected False most of the time -- augmentation re-rolled per call)")
        else:
            assert identical, f"{name}: repeated fetch of the SAME index produced DIFFERENT tensors " \
                               f"-- augmentation is leaking into a non-train split!"
            print(f"[behavior] {name}: repeated fetch identical={identical} (expected True) -- OK")

    print("\nAll checks passed: augmentation is confirmed train-only.")


if __name__ == "__main__":
    main()
