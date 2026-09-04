"""
Objectively cross-reference Grad-CAM hotspots against VISTA3D organ labels:
maps the top-5%-intensity CAM voxels (in the 224^3 preprocessed grid) back
through the exact crop+resize transform chain to physical mm space, then
into the independently-computed VISTA3D segmentation grid's own voxel space,
and looks up which organ label sits there.

No cached segmentation label map exists on disk (only the bbox extremes were
ever persisted in roi_bboxes.csv) -- segmentation is recomputed fresh here
via roi_localizer.segment_organ_bbox_mm(return_label_map=True), same as the
original ROI-cropping step, since re-running is the only way to get it.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import EXTENDED_FOV_THRESHOLD_MM, PneumoDataset, _load_reoriented_resampled  # noqa: E402
from gradcam import GradCAM3D, load_finetuned_model_for_gradcam  # noqa: E402
from roi_localizer import ROI_ORGAN_IDS, segment_organ_bbox_mm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
ROI_CSV = os.path.join(ROOT, "scripts", "roi_bboxes.csv")
CHECKPOINT = os.path.join(ROOT, "runs", "finetune_20260902_140547", "checkpoints", "best.pt")
PATCH_SIZE = 224

ORGAN_NAMES = {
    1: "liver", 2: "kidney", 3: "spleen", 4: "pancreas", 5: "right kidney",
    7: "IVC", 8: "right adrenal", 9: "left adrenal", 10: "gallbladder",
    11: "esophagus", 12: "stomach", 13: "duodenum", 14: "left kidney",
    15: "bladder", 17: "portal/splenic vein", 18: "rectum", 19: "small bowel", 62: "colon",
    0: "background/none",
}


def get_cropped_affine_and_shape(filepath, extent_z_mm, roi_row):
    """Reproduce dataset.py's reorient+resample(+crop) pipeline, but return the
    CORRECT affine of the resulting grid (before the final resize-to-224),
    since MetaTensor's plain slicing does not update the affine origin."""
    data = _load_reoriented_resampled(filepath)
    affine = np.asarray(data.affine, dtype=np.float64)
    shape = np.array(data.shape[1:])

    if extent_z_mm >= EXTENDED_FOV_THRESHOLD_MM:
        bbox_min_mm = (roi_row["bbox_min_x"], roi_row["bbox_min_y"], roi_row["bbox_min_z"])
        bbox_max_mm = (roi_row["bbox_max_x"], roi_row["bbox_max_y"], roi_row["bbox_max_z"])
        inv = np.linalg.inv(affine)
        corners_mm = np.array([[bbox_min_mm[0], bbox_min_mm[1], bbox_min_mm[2], 1],
                                [bbox_max_mm[0], bbox_max_mm[1], bbox_max_mm[2], 1]])
        corners_vox = (inv @ corners_mm.T).T[:, :3]
        vox_min = np.floor(corners_vox.min(axis=0)).astype(int)
        vox_max = np.ceil(corners_vox.max(axis=0)).astype(int)
        vox_min = np.clip(vox_min, 0, shape - 1)
        vox_max = np.clip(vox_max, vox_min + 1, shape)
    else:
        vox_min = np.array([0, 0, 0])
        vox_max = shape

    cropped_shape = vox_max - vox_min
    cropped_affine = affine.copy()
    origin_mm = affine @ np.array([vox_min[0], vox_min[1], vox_min[2], 1.0])
    cropped_affine[:3, 3] = origin_mm[:3]

    return cropped_affine, cropped_shape


def resized_idx_to_cropped_idx(resized_idx, cropped_shape, patch_size=PATCH_SIZE):
    """Invert F.interpolate(..., mode='trilinear', align_corners=False)'s
    index mapping: input_idx = (output_idx + 0.5) * (in_size/out_size) - 0.5"""
    resized_idx = np.asarray(resized_idx, dtype=np.float64)
    scale = np.asarray(cropped_shape, dtype=np.float64) / patch_size
    cropped_idx = (resized_idx + 0.5) * scale - 0.5
    return cropped_idx


def lookup_organs_for_cam(cam: np.ndarray, cropped_affine, cropped_shape, seg_label_map, seg_affine,
                           top_frac: float = 0.05):
    """Top-`top_frac` CAM voxels -> physical mm -> seg grid voxel -> organ label."""
    flat = cam.flatten()
    n_top = max(1, int(len(flat) * top_frac))
    top_flat_idx = np.argpartition(flat, -n_top)[-n_top:]
    top_idx_3d = np.array(np.unravel_index(top_flat_idx, cam.shape)).T  # [N, 3] in resized (224) grid

    cropped_idx = resized_idx_to_cropped_idx(top_idx_3d, cropped_shape)  # [N, 3]
    cropped_idx_h = np.concatenate([cropped_idx, np.ones((len(cropped_idx), 1))], axis=1)
    physical_mm = (cropped_affine @ cropped_idx_h.T).T[:, :3]  # [N, 3]

    seg_inv = np.linalg.inv(seg_affine)
    physical_mm_h = np.concatenate([physical_mm, np.ones((len(physical_mm), 1))], axis=1)
    seg_vox = (seg_inv @ physical_mm_h.T).T[:, :3]
    seg_vox_rounded = np.round(seg_vox).astype(int)

    seg_shape = np.array(seg_label_map.shape)
    in_bounds = np.all((seg_vox_rounded >= 0) & (seg_vox_rounded < seg_shape), axis=1)

    labels = []
    for i, ok in enumerate(in_bounds):
        if not ok:
            labels.append(-1)  # out of segmentation FOV entirely
            continue
        x, y, z = seg_vox_rounded[i]
        labels.append(int(seg_label_map[x, y, z]))

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    organ_counts = {int(u): int(c) for u, c in zip(unique, counts)}
    return organ_counts, len(labels)


def analyze_volume(filepath, model, cam4, cam3):
    audit = pd.read_csv(AUDIT_CSV)
    roi_df = pd.read_csv(ROI_CSV)
    row = audit[audit.filepath == filepath].iloc[0]
    roi_row = None
    match = roi_df[roi_df.filepath == filepath]
    if len(match) and match.iloc[0].get("status") == "ok":
        roi_row = match.iloc[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PneumoDataset([filepath], patch_size=(PATCH_SIZE,) * 3)
    x, y, _ = ds[0]
    x = x.unsqueeze(0).to(device)

    cam4_map, _, prob = cam4(x, target_size=(PATCH_SIZE,) * 3)
    cam3_map, _, _ = cam3(x, target_size=(PATCH_SIZE,) * 3)

    cropped_affine, cropped_shape = get_cropped_affine_and_shape(filepath, row["extent_z_mm"], roi_row)

    print(f"\nRunning VISTA3D segmentation on {os.path.basename(filepath)} for organ lookup...")
    seg_result = segment_organ_bbox_mm(filepath, organ_ids=ROI_ORGAN_IDS, return_label_map=True)
    seg_label_map = seg_result["label_map"]
    seg_affine = seg_result["label_map_affine"]

    organs4, n4 = lookup_organs_for_cam(cam4_map, cropped_affine, cropped_shape, seg_label_map, seg_affine)
    organs3, n3 = lookup_organs_for_cam(cam3_map, cropped_affine, cropped_shape, seg_label_map, seg_affine)

    def fmt(organ_counts, n):
        items = sorted(organ_counts.items(), key=lambda kv: -kv[1])
        parts = []
        for label_id, count in items:
            name = "out-of-segmentation-FOV" if label_id == -1 else ORGAN_NAMES.get(label_id, f"id{label_id}")
            parts.append(f"{name} ({100*count/n:.0f}%)")
        return ", ".join(parts)

    print(f"  prob={prob:.3f}")
    print(f"  stage4 (top 5% of {cam4_map.size} voxels, n={n4}): {fmt(organs4, n4)}")
    print(f"  stage3 (top 5% of {cam3_map.size} voxels, n={n3}): {fmt(organs3, n3)}")

    top4 = max(organs4.items(), key=lambda kv: kv[1])
    top3 = max(organs3.items(), key=lambda kv: kv[1])
    agree = top4[0] == top3[0]
    print(f"  dominant organ -- stage4: {ORGAN_NAMES.get(top4[0], top4[0])}, "
          f"stage3: {ORGAN_NAMES.get(top3[0], top3[0])} -- {'AGREE' if agree else 'DISAGREE'}")

    return {"filepath": filepath, "prob": prob, "organs4": organs4, "organs3": organs3,
            "dominant4": top4[0], "dominant3": top3[0], "agree": agree}


def main():
    tp_files = [
        "merged/Pneumo_Positive/20251028201-1.nii.gz",
        "merged/Pneumo_Positive/20251028360-1.nii.gz",
        "merged/Pneumo_Positive/20251028388-1.nii.gz",
    ]
    tp_files = [f.replace("/", os.sep) for f in tp_files]

    model, ckpt = load_finetuned_model_for_gradcam(CHECKPOINT, unfreeze_stages=(4,))
    cam4 = GradCAM3D(model, layer_name="stage4")
    cam3 = GradCAM3D(model, layer_name="stage3")

    results = []
    for fp in tp_files:
        # resolve to full path matching audit_results.csv's filepath format
        audit = pd.read_csv(AUDIT_CSV)
        matches = audit[audit.filename == os.path.basename(fp)]
        assert len(matches) == 1, f"expected 1 match for {fp}, got {len(matches)}"
        full_fp = matches.iloc[0]["filepath"]
        results.append(analyze_volume(full_fp, model, cam4, cam3))

    cam4.remove()
    cam3.remove()

    print("\n=== SUMMARY: dominant organ at CAM hotspot, all TP cases ===")
    for r in results:
        print(f"  {os.path.basename(r['filepath'])}: prob={r['prob']:.3f}, "
              f"stage4->{ORGAN_NAMES.get(r['dominant4'], r['dominant4'])}, "
              f"stage3->{ORGAN_NAMES.get(r['dominant3'], r['dominant3'])}, "
              f"{'AGREE' if r['agree'] else 'DISAGREE'}")


if __name__ == "__main__":
    main()
