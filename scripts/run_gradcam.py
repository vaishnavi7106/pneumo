"""
Run 3D Grad-CAM on the fine-tuned checkpoint (epoch 5,
runs/finetune_20260902_140547) for a handful of true-positive cases (primary
ask), plus false negatives/positives if time allows. Uses the VAL set to
pick examples (not test -- test was already used once for the final
reported numbers and shouldn't be touched again just to pick pictures).

For each selected volume: runs Grad-CAM at stage4 (primary, 8^3 -> coarse,
most class-specific) and stage3 (secondary, 16^3 -> finer, less
class-specific), renders axial/coronal/sagittal panels through the heatmap's
peak voxel (not just the volume center, so the hot region is actually
visible), CT in grayscale with the CAM overlaid as a semi-transparent
colormap.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PneumoDataset  # noqa: E402
from gradcam import GradCAM3D, load_finetuned_model_for_gradcam  # noqa: E402
from split import patient_level_split  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT, "runs", "finetune_20260902_140547", "checkpoints", "best.pt")
THRESHOLD = 0.63  # tuned on val for this exact checkpoint (see threshold_tuning.py run)
OUT_DIR = os.path.join(ROOT, "gradcam_outputs")


def get_val_predictions(model, val_files, device):
    ds = PneumoDataset(val_files, patch_size=(224, 224, 224))
    records = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, path = ds[i]
            x = x.unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logit = model(x)
            prob = torch.sigmoid(logit.float()).item()
            records.append({"filepath": path, "label": int(y.item()), "prob": prob})
    return pd.DataFrame(records)


def render_case(volume_tensor, cam_stage4, cam_stage3, title, out_path):
    """volume_tensor: [D,H,W] numpy in [0,1]. cam_*: [D,H,W] numpy in [0,1]."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row_idx, (cam, cam_name) in enumerate([(cam_stage4, "stage4 (8^3, coarse)"),
                                                 (cam_stage3, "stage3 (16^3, finer)")]):
        peak = np.unravel_index(np.argmax(cam), cam.shape)  # (d, h, w) = (x,y,z) after RAS-oriented resize
        for col_idx, (dim, label) in enumerate([(2, "axial"), (1, "coronal"), (0, "sagittal")]):
            idx = peak[dim]
            if dim == 2:
                ct_slice = volume_tensor[:, :, idx].T
                cam_slice = cam[:, :, idx].T
            elif dim == 1:
                ct_slice = volume_tensor[:, idx, :].T
                cam_slice = cam[:, idx, :].T
            else:
                ct_slice = volume_tensor[idx, :, :].T
                cam_slice = cam[idx, :, :].T

            ax = axes[row_idx, col_idx]
            ax.imshow(ct_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.imshow(cam_slice, cmap="jet", origin="lower", alpha=0.45 * (cam_slice > 0.15), vmin=0, vmax=1)
            ax.set_title(f"{cam_name} {label} (idx={idx})")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_finetuned_model_for_gradcam(CHECKPOINT, unfreeze_stages=(4,))
    print(f"Loaded checkpoint: {CHECKPOINT}")
    print(f"  epoch={ckpt.get('epoch')}, score={ckpt.get('selection_score')}")

    config = ckpt["config"]
    splits, _ = patient_level_split(seed=config["split_seed"])

    print("\nComputing val-set predictions to select TP/FN/FP examples...")
    preds = get_val_predictions(model, splits["val"], device)
    preds["pred"] = (preds["prob"] >= THRESHOLD).astype(int)

    tp = preds[(preds.label == 1) & (preds.pred == 1)].sort_values("prob", ascending=False)
    fn = preds[(preds.label == 1) & (preds.pred == 0)].sort_values("prob")
    fp = preds[(preds.label == 0) & (preds.pred == 1)].sort_values("prob", ascending=False)

    print(f"\nVal set: {len(preds)} total. TP={len(tp)}, FN={len(fn)}, FP={len(fp)} at threshold={THRESHOLD}")

    n_tp = min(4, len(tp))
    n_fn = min(2, len(fn))
    n_fp = min(2, len(fp))
    selected = [(row, "TP") for _, row in tp.head(n_tp).iterrows()]
    selected += [(row, "FN") for _, row in fn.head(n_fn).iterrows()]
    selected += [(row, "FP") for _, row in fp.head(n_fp).iterrows()]

    print(f"\nSelected {len(selected)} volumes: {n_tp} TP, {n_fn} FN, {n_fp} FP")

    cam4 = GradCAM3D(model, layer_name="stage4")
    cam3 = GradCAM3D(model, layer_name="stage3")

    ds_lookup = PneumoDataset([r["filepath"] for r, _ in selected], patch_size=(224, 224, 224))

    results = []
    for row, category in selected:
        filepath = row["filepath"]
        filename = os.path.basename(filepath)
        idx = ds_lookup.filepaths.index(filepath)
        x, y, _ = ds_lookup[idx]
        x = x.unsqueeze(0).to(device)

        cam4_map, logit4, prob4 = cam4(x, target_size=(224, 224, 224))
        cam3_map, logit3, prob3 = cam3(x, target_size=(224, 224, 224))

        assert abs(prob4 - prob3) < 1e-4, "prob should be identical regardless of which layer is hooked"
        assert not np.isnan(cam4_map).any() and not np.isnan(cam3_map).any()

        vol = x[0, 0].detach().cpu().numpy()
        title = (f"{filename} | true_label={'positive' if row['label']==1 else 'negative'} | "
                 f"category={category} | prob={row['prob']:.3f} (threshold={THRESHOLD})")
        out_path = os.path.join(OUT_DIR, f"{category}__{filename.replace('.nii.gz', '')}.png")
        render_case(vol, cam4_map, cam3_map, title, out_path)

        print(f"[{category}] {filename}: prob={row['prob']:.3f}, "
              f"cam4 peak voxel={np.unravel_index(np.argmax(cam4_map), cam4_map.shape)}, "
              f"cam3 peak voxel={np.unravel_index(np.argmax(cam3_map), cam3_map.shape)} "
              f"-> saved {out_path}")
        results.append({"filename": filename, "category": category, "prob": row["prob"],
                         "out_path": out_path})

    cam4.remove()
    cam3.remove()

    print(f"\nAll {len(results)} Grad-CAM visualizations saved to {OUT_DIR}")
    return results


if __name__ == "__main__":
    main()
