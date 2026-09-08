"""
Stage-2 data prep: extract every connected component from the (corrected,
closing+erode-back) air-mask channel across all 23 merged/GT/ volumes, label
each component positive/negative by voxel overlap with the real GT mask,
compute shape features, and create a volume-level train/holdout split -- all
components from a given volume stay entirely on one side, same leakage rule
used everywhere else in this project.

NOTE: there is no earlier saved split from the closing-radius sweep to reuse
-- eval_air_mask_vs_gt.py evaluated all 23 volumes without holding any out,
so no such split exists. This creates ONE new deterministic volume-level
split (saved to component_volume_split.json) and reuses it on every future
run of this script, so results stay comparable across script re-runs.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from component_features import compute_component_features, extract_components  # noqa: E402
from dataset import BODY_MASK_HU_THRESHOLD  # noqa: E402
from eval_air_mask_vs_gt import compute_body_mask  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(ROOT, "merged", "GT")
CT_DIR = os.path.join(ROOT, "merged", "Pneumo_Positive")
OUT_DIR = os.path.join(ROOT, "scripts", "component_classifier")
AIR_THRESHOLD = -600.0
CLOSING_ITERS = 3
ERODE_ITERS = 1
HOLDOUT_FRAC = 0.3
SPLIT_SEED = 17


def make_or_load_split(gt_files):
    split_path = os.path.join(OUT_DIR, "component_volume_split.json")
    if os.path.exists(split_path):
        print(f"Reusing existing split from {split_path}")
        with open(split_path) as f:
            return json.load(f)

    print(f"No existing component-level split found -- creating a new one "
          f"(seed={SPLIT_SEED}, {HOLDOUT_FRAC:.0%} holdout) and saving to {split_path}")
    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = list(gt_files)
    rng.shuffle(shuffled)
    n_holdout = max(1, round(len(shuffled) * HOLDOUT_FRAC))
    holdout = sorted(shuffled[:n_holdout])
    train = sorted(shuffled[n_holdout:])
    split = {"train": train, "holdout": holdout}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    return split


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gt_files = sorted(os.listdir(GT_DIR))
    split = make_or_load_split(gt_files)
    volume_to_fold = {fn: ("train" if fn in split["train"] else "holdout") for fn in gt_files}
    print(f"\nVolume split: {len(split['train'])} train, {len(split['holdout'])} holdout")
    print(f"  train:   {split['train']}")
    print(f"  holdout: {split['holdout']}\n")

    rows = []
    pbar = tqdm(gt_files, desc="volumes", unit="vol", file=sys.stdout)
    for fname in pbar:
        pbar.set_postfix_str(fname)
        ct_path = os.path.join(CT_DIR, fname)
        gt_path = os.path.join(GT_DIR, fname)
        img = nib.load(ct_path)
        raw_hu = img.get_fdata(dtype=np.float32)
        spacing = img.header.get_zooms()[:3]
        gt = nib.load(gt_path).get_fdata().astype(bool)

        body_mask = compute_body_mask(torch.as_tensor(raw_hu), BODY_MASK_HU_THRESHOLD,
                                       CLOSING_ITERS, ERODE_ITERS).numpy().astype(bool)
        air_mask = (raw_hu < AIR_THRESHOLD) & body_mask

        labeled, kept_ids = extract_components(air_mask)
        n_pos, n_neg = 0, 0
        for cid in kept_ids:
            comp_mask = labeled == cid
            overlaps_gt = bool((comp_mask & gt).any())
            feats = compute_component_features(comp_mask, body_mask, spacing)
            feats.update({"filename": fname, "fold": volume_to_fold[fname],
                          "component_id": int(cid), "label": int(overlaps_gt)})
            rows.append(feats)
            n_pos += int(overlaps_gt)
            n_neg += int(not overlaps_gt)
        tqdm.write(f"{fname} ({volume_to_fold[fname]:>7s}): "
                   f"{len(kept_ids)} components -> {n_pos} positive, {n_neg} negative")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "component_dataset.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n=== TOTAL: {len(df)} components across {len(gt_files)} volumes ===")
    print(df.groupby("fold")["label"].value_counts().to_string())

    n_pos_total = int(df.label.sum())
    n_neg_total = int((df.label == 0).sum())
    if n_pos_total < 15 or n_neg_total < 30:
        print(f"\nWARNING: small sample ({n_pos_total} positive, {n_neg_total} negative components "
              f"total). Any classifier trained on this should be treated as a directional signal, "
              f"not a reliable estimate -- say so plainly when reporting results.")

    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
