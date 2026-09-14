"""
Same QC panel as qc_air_mask.py, but for the two cases with the HIGHEST
GT-verified air-mask IoU/recall (from air_mask_qc_extended/gt_eval_closing3_erode1.csv)
-- i.e. examples where the mask is doing a genuinely good job, as a
counterpoint to the cases where pockets were visibly missed.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PneumoDataset  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
GT_EVAL_CSV = os.path.join(ROOT, "scripts", "air_mask_qc_extended", "gt_eval_closing3_erode1.csv")
OUT_DIR = os.path.join(ROOT, "scripts", "air_mask_qc")
AIR_MASK_THRESHOLD = -600.0

BEST_CASES = ["20251028060-1.nii.gz", "20251028077-1.nii.gz"]


def render_panel(intensity, mask, title, out_path):
    c = [s // 2 for s in intensity.shape]
    views = [
        (intensity[c[0], :, :].T, mask[c[0], :, :].T, "sagittal"),
        (intensity[:, c[1], :].T, mask[:, c[1], :].T, "coronal"),
        (intensity[:, :, c[2]].T, mask[:, :, c[2]].T, "axial"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for col, (ct_slice, mask_slice, name) in enumerate(views):
        axes[0, col].imshow(ct_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0, col].set_title(f"CT intensity {name}")
        axes[0, col].axis("off")

        axes[1, col].imshow(ct_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, col].imshow(mask_slice, cmap="autumn", origin="lower", alpha=0.5 * (mask_slice > 0.1),
                             vmin=0, vmax=1)
        axes[1, col].set_title(f"+ air mask (<{AIR_MASK_THRESHOLD:.0f} HU) {name}")
        axes[1, col].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    audit = pd.read_csv(AUDIT_CSV)
    audit["basename"] = audit["filepath"].apply(os.path.basename)
    gt_eval = pd.read_csv(GT_EVAL_CSV).set_index("filename")

    fps = []
    for b in BEST_CASES:
        row = audit[audit["basename"] == b]
        fps.append(row.iloc[0]["filepath"])

    ds = PneumoDataset(fps, patch_size=(160, 160, 160),
                        air_mask_threshold=AIR_MASK_THRESHOLD, use_cache=False)

    for i, (fp, basename) in enumerate(zip(fps, BEST_CASES)):
        x, y, _ = ds[i]
        intensity = x[0].numpy()
        mask = x[1].numpy()

        gt_row = gt_eval.loc[basename]
        title = (f"{basename} label=positive | GT-verified IoU={gt_row['iou']:.3f}, "
                 f"recall={gt_row['recall']:.3f}, precision={gt_row['precision']:.3f}")
        out_path = os.path.join(OUT_DIR, f"air_mask_qc_best__{basename.replace('.nii.gz', '')}.png")
        render_panel(intensity, mask, title, out_path)
        print(f"{basename}: IoU={gt_row['iou']:.3f}, recall={gt_row['recall']:.3f} -> saved {out_path}")


if __name__ == "__main__":
    main()
