"""
Visual sanity check: original native CT vs. the new fixed-FOV preprocessed
tensor (adaptive ROI crop + constant 3mm/voxel spacing + pad/crop to 224^3,
NO resize), side by side, all 3 views -- so we can actually SEE whether this
pipeline looks reasonable before trusting the numbers.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PneumoDataset, FIXED_FOV_PATCH_SIZE, FIXED_FOV_SPACING, HU_A_MIN, HU_A_MAX  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
ADAPTIVE_ROI_CSV = os.path.join(ROOT, "scripts", "roi_bboxes_adaptive.csv")
OUT_DIR = os.path.join(ROOT, "scripts", "air_mask_qc")


def render_case(fp, label, ds):
    idx = ds.filepaths.index(fp)
    x, y, _ = ds[idx]
    processed = x[0].numpy()  # intensity channel, [224,224,224]

    # original, raw, native-resolution CT (no preprocessing at all)
    orig_img = nib.as_closest_canonical(nib.load(fp))
    orig = orig_img.get_fdata(dtype=np.float32)
    orig_scaled = np.clip(orig, HU_A_MIN, HU_A_MAX)
    orig_scaled = (orig_scaled - HU_A_MIN) / (HU_A_MAX - HU_A_MIN)

    oc = [s // 2 for s in orig_scaled.shape]
    pc = [s // 2 for s in processed.shape]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    views = ["sagittal", "coronal", "axial"]

    orig_slices = [orig_scaled[oc[0], :, :].T, orig_scaled[:, oc[1], :].T, orig_scaled[:, :, oc[2]].T]
    proc_slices = [processed[pc[0], :, :].T, processed[:, pc[1], :].T, processed[:, :, pc[2]].T]

    for col, name in enumerate(views):
        axes[0, col].imshow(orig_slices[col], cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0, col].set_title(f"ORIGINAL (native, shape={orig_scaled.shape}) {name}")
        axes[0, col].axis("off")

        axes[1, col].imshow(proc_slices[col], cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, col].set_title(f"PREPROCESSED (fixed-FOV, {FIXED_FOV_PATCH_SIZE[0]}^3 @ "
                                f"{FIXED_FOV_SPACING[0]}mm) {name}")
        axes[1, col].axis("off")

    filename = os.path.basename(fp)
    fig.suptitle(f"{filename}  label={label}")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"fixed_fov_vs_original__{label}__{filename.replace('.nii.gz', '')}.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[{label}] {filename} -> saved {out_path}")
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    audit = pd.read_csv(AUDIT_CSV)

    pos_fp = audit[audit.label == "positive"].sample(n=1, random_state=3).iloc[0]["filepath"]
    neg_fp = audit[audit.label == "negative"].sample(n=1, random_state=3).iloc[0]["filepath"]

    ds = PneumoDataset([pos_fp, neg_fp], patch_size=FIXED_FOV_PATCH_SIZE, use_cache=True,
                        fixed_fov=True, fixed_fov_spacing=FIXED_FOV_SPACING,
                        roi_bbox_csv=ADAPTIVE_ROI_CSV)

    out_paths = []
    out_paths.append(render_case(pos_fp, "positive", ds))
    out_paths.append(render_case(neg_fp, "negative", ds))
    return out_paths


if __name__ == "__main__":
    main()
