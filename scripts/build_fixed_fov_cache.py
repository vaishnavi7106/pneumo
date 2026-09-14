"""
Build the on-disk cache for the new fixed-FOV pipeline (adaptive ROI crop
via roi_bboxes_adaptive.csv + fixed 3mm/voxel spacing + pad/crop to 224^3,
no resize -- see dataset.py's fixed_fov=True path) across ALL 363 volumes,
for both configs we'll actually want to compare next: baseline (1-channel)
and air-mask (2-channel, raw -600 HU threshold, the only variant that's
given a real improvement so far this month).

PneumoDataset caches lazily inside __getitem__ (writes to cache_dir on first
access, reused after) -- this script just forces a full pass over every
volume once, via a DataLoader with multiple workers for speed, so the cache
is fully populated before any actual training run touches it (avoids
eating the cache-miss cost during the first training epoch instead).
"""
import argparse
import os
import sys
import time

import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FIXED_FOV_PATCH_SIZE, FIXED_FOV_SPACING, PneumoDataset  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
ADAPTIVE_ROI_CSV = os.path.join(ROOT, "scripts", "roi_bboxes_adaptive.csv")

CONFIGS = [
    {"name": "baseline_1ch", "air_mask_threshold": None},
    {"name": "air_mask_2ch", "air_mask_threshold": -600.0},
]


def build_cache_for_config(config, filepaths, num_workers):
    print(f"\n{'='*70}\n=== Building cache: {config['name']} ===\n{'='*70}")
    ds = PneumoDataset(
        filepaths, patch_size=FIXED_FOV_PATCH_SIZE, use_cache=True,
        air_mask_threshold=config["air_mask_threshold"],
        fixed_fov=True, fixed_fov_spacing=FIXED_FOV_SPACING,
        roi_bbox_csv=ADAPTIVE_ROI_CSV,
    )
    print(f"Cache dir: {ds.cache_dir}")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)

    t0 = time.time()
    n_done = 0
    for i, (x, y, fp) in enumerate(loader):
        n_done += 1
        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(ds) - n_done) / rate if rate > 0 else float("nan")
            print(f"  [{n_done}/{len(ds)}] shape={tuple(x.shape)} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    print(f"Done: {n_done}/{len(ds)} volumes cached in {time.time()-t0:.0f}s -> {ds.cache_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--configs", nargs="+", default=[c["name"] for c in CONFIGS],
                    help="which named configs to build (default: all)")
    args = p.parse_args()

    audit = pd.read_csv(AUDIT_CSV)
    filepaths = audit["filepath"].tolist()
    print(f"{len(filepaths)} total volumes to cache")

    for config in CONFIGS:
        if config["name"] not in args.configs:
            continue
        build_cache_for_config(config, filepaths, args.num_workers)


if __name__ == "__main__":
    main()
