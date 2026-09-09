"""
Visual QC for the boundary-distance channel (see dataset.py's
_compute_boundary_distance_channel / compare_boundary_distance_fold4.py):
renders the CT intensity channel alongside the computed distance-to-body-wall
field, same panel style as qc_air_mask.py. Confirms the channel is actually a
sensible geometric prior -- low near the body surface, high toward the
center, correctly zero outside the body -- rather than something broken
(e.g. spatially misaligned, inverted, or saturated everywhere).

Picks one positive (pneumoperitoneum) and one negative case, same sampling
(random_state=3) as qc_air_mask.py so the exact same two volumes are used --
lets the two channels be visually compared side by side for the same cases.
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
from dataset import BOUNDARY_DISTANCE_CAP_MM, PneumoDataset  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
OUT_DIR = os.path.join(ROOT, "scripts", "boundary_distance_qc")


def render_panel(intensity, dist, title, out_path):
    c = [s // 2 for s in intensity.shape]
    views = [
        (intensity[c[0], :, :].T, dist[c[0], :, :].T, "sagittal"),
        (intensity[:, c[1], :].T, dist[:, c[1], :].T, "coronal"),
        (intensity[:, :, c[2]].T, dist[:, :, c[2]].T, "axial"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for col, (ct_slice, dist_slice, name) in enumerate(views):
        axes[0, col].imshow(ct_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0, col].set_title(f"CT intensity {name}")
        axes[0, col].axis("off")

        im = axes[1, col].imshow(dist_slice, cmap="viridis", origin="lower", vmin=0, vmax=1)
        axes[1, col].set_title(f"boundary distance (0-{BOUNDARY_DISTANCE_CAP_MM:.0f}mm) {name}")
        axes[1, col].axis("off")

    fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, pad=0.02, label="normalized distance")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    audit = pd.read_csv(AUDIT_CSV)

    pos_fp = audit[audit.label == "positive"].sample(n=1, random_state=3).iloc[0]["filepath"]
    neg_fp = audit[audit.label == "negative"].sample(n=1, random_state=3).iloc[0]["filepath"]

    ds = PneumoDataset([pos_fp, neg_fp], patch_size=(160, 160, 160),
                        boundary_distance_channel=True, use_cache=False)

    for i, (fp, label) in enumerate([(pos_fp, "positive"), (neg_fp, "negative")]):
        x, y, _ = ds[i]
        assert x.shape[0] == 2, f"expected 2 channels, got {x.shape[0]}"
        intensity = x[0].numpy()
        dist = x[1].numpy()

        filename = os.path.basename(fp)
        frac_zero = (dist < 0.01).mean()
        mean_inside = dist[dist > 0.01].mean() if (dist > 0.01).any() else float("nan")
        title = (f"{filename} label={label} | outside-body fraction (dist<0.01)={frac_zero:.3f}, "
                 f"mean inside-body distance={mean_inside:.3f}")
        out_path = os.path.join(OUT_DIR, f"boundary_distance_qc__{label}__{filename.replace('.nii.gz', '')}.png")
        render_panel(intensity, dist, title, out_path)
        print(f"[{label}] {filename}: outside-body fraction={frac_zero:.3f}, "
              f"mean inside-body distance={mean_inside:.3f} -> saved {out_path}")

    print(f"\nAll QC panels saved to {OUT_DIR}. Inspect for:")
    print("  - distance field is 0 outside the body, rising smoothly toward the body's interior")
    print("  - spatially aligns with the intensity channel (no offset/misregistration)")
    print("  - no obvious inversion (center should be brightest/highest, surface darkest/lowest)")


if __name__ == "__main__":
    main()
