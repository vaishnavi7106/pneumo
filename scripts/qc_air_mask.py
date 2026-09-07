"""
Visual QC for the air-mask channel: renders the CT intensity channel with the
computed air mask (HU < threshold) overlaid, same panel style as the earlier
ROI-crop and augmentation QC. Confirms the mask actually highlights plausible
air-density regions (bowel gas, free air if present, background air outside
the body) rather than something broken (e.g. an all-zero/all-one mask, or a
mask that's spatially misaligned with the intensity channel).

Picks one positive (pneumoperitoneum) and one negative case so the positive
case's free-air pattern -- the actual clinical signal this channel is meant
to help the model see -- can be checked directly against the mask.
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
OUT_DIR = os.path.join(ROOT, "scripts", "air_mask_qc")
AIR_MASK_THRESHOLD = -600.0


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

    pos_fp = audit[audit.label == "positive"].sample(n=1, random_state=3).iloc[0]["filepath"]
    neg_fp = audit[audit.label == "negative"].sample(n=1, random_state=3).iloc[0]["filepath"]

    ds = PneumoDataset([pos_fp, neg_fp], patch_size=(160, 160, 160),
                        air_mask_threshold=AIR_MASK_THRESHOLD, use_cache=False)

    for i, (fp, label) in enumerate([(pos_fp, "positive"), (neg_fp, "negative")]):
        x, y, _ = ds[i]
        assert x.shape[0] == 2, f"expected 2 channels, got {x.shape[0]}"
        intensity = x[0].numpy()
        mask = x[1].numpy()

        filename = os.path.basename(fp)
        frac_air = (mask > 0.5).mean()
        title = f"{filename} label={label} | air-mask fraction (>0.5)={frac_air:.3f}"
        out_path = os.path.join(OUT_DIR, f"air_mask_qc__{label}__{filename.replace('.nii.gz', '')}.png")
        render_panel(intensity, mask, title, out_path)
        print(f"[{label}] {filename}: air fraction (mask>0.5)={frac_air:.3f} -> saved {out_path}")

    print(f"\nAll QC panels saved to {OUT_DIR}. Inspect for:")
    print("  - mask highlights bowel gas / background air (dark regions in the CT) -- expected")
    print("  - mask spatially aligns with the intensity channel (no offset/misregistration)")
    print("  - the positive case's mask should include the free-air region typical of pneumoperitoneum")


if __name__ == "__main__":
    main()
