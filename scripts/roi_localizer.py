"""
Step 7: ROI localization for extended-FOV volumes using VISTA3D's full
network (image_encoder + point_head + class_head) loaded directly via
monai.networks.nets.vista3d132(), run in "everything"-style auto-segmentation
mode (class_vector prompts, no point prompts) via sliding-window inference.

Deliberate deviation from the bundle's full ~116-class "everything_labels"
set: we only request the abdominal organ labels we actually need (18 ids),
not the full label set. This is a valid use of the same API (class_vector
can be any subset of the 132 supported ids) and cuts sliding-window compute/
memory roughly 6x, which matters on a 12GB card running full-body volumes.

Output: axis-aligned bounding box of the requested organs, in RAS physical
mm space (not voxel space), with a margin -- physical space is shared across
reorient/resample operations, so this bbox can be reused directly against
any resampled grid of the same volume during actual training preprocessing.
"""
import os

import numpy as np
import torch
from monai.apps.vista3d.transforms import VistaPostTransformd
from monai.data import MetaTensor
from monai.inferers import SlidingWindowInfererAdapt
from monai.networks.nets import vista3d132
from monai.transforms import CropForeground, Orientation, ScaleIntensityRange, Spacing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_CKPT = os.path.join(ROOT, "vista3d", "models", "model.pt")

HU_A_MIN = -963.8247715525971
HU_A_MAX = 1053.678477684517
RESAMPLE_SPACING = (1.5, 1.5, 1.5)
PATCH_SIZE = (128, 128, 128)

# organ IDs from metadata.json (see PROJECT_BRIEFING.md), used to define the
# abdominal ROI for cropping extended-FOV volumes
ROI_ORGAN_IDS = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 62]

# "core" abdominal organs used ONLY for the z-centroid computation (not for
# x/y bbox, and not excluded from segmentation/found_labels) -- deliberately
# excludes rectum (18) and colon (62) because their union bbox with the rest
# pulls the crop deep into the pelvis, and esophagus (11)/bladder(15)/portal
# vein(17)/IVC(7) are similarly peripheral-to-the-core-mass. This centroid
# should sit in the upper-mid abdomen where the bulk of solid organs are.
CORE_ORGAN_IDS_FOR_CENTROID = [1, 3, 4, 5, 8, 9, 10, 12, 13, 14, 19]

# air/background HU cutoff used only to speed up sliding-window inference by
# skipping obvious full-body-scan air padding before segmenting -- NOT used
# for the organ bbox itself (that comes from the network's own output)
FOREGROUND_HU_CUTOFF = -500.0
FOREGROUND_MARGIN_VOX = 10

_MODEL = None


def get_vista3d_model() -> torch.nn.Module:
    """Load the full VISTA3D network (image_encoder + point_head + class_head)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    model = vista3d132(in_channels=1)
    sd = torch.load(BUNDLE_CKPT, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    _MODEL = model
    return model


def _load_as_meta_tensor(filepath: str) -> MetaTensor:
    import nibabel as nib

    img = nib.load(filepath)
    data = img.get_fdata(dtype=np.float32)[None]  # [1, X, Y, Z]
    affine = torch.as_tensor(img.affine, dtype=torch.float64)
    return MetaTensor(torch.as_tensor(data, dtype=torch.float32), affine=affine)


def segment_organ_bbox_mm(filepath: str, organ_ids=ROI_ORGAN_IDS, margin_mm: float = 20.0,
                           overlap: float = 0.25, verbose: bool = False, return_label_map: bool = False):
    """
    Run VISTA3D auto-segmentation on `filepath` for the given organ_ids and
    return the axis-aligned bounding box (with margin) of their union, in
    RAS physical mm coordinates.

    Returns a dict: {
        'bbox_min_mm': (x,y,z), 'bbox_max_mm': (x,y,z),
        'found_labels': set of organ ids actually present in the prediction,
        'resampled_shape': shape of the volume the network actually saw,
        'status': 'ok' | 'no_foreground_found',
    }
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_vista3d_model()

    data = _load_as_meta_tensor(filepath)  # [1, X, Y, Z], raw HU

    spacing_tf = Spacing(pixdim=RESAMPLE_SPACING, mode="bilinear")
    data = spacing_tf(data)  # still raw HU, MetaTensor carries affine

    # crop obvious air padding before sliding window (speed only, generous margin,
    # organs are always >> -500 HU so this cannot clip real anatomy)
    cropper = CropForeground(select_fn=lambda x: x > FOREGROUND_HU_CUTOFF,
                              margin=FOREGROUND_MARGIN_VOX, allow_smaller=True)
    data = cropper(data)

    resampled_shape = tuple(data.shape[1:])
    if verbose:
        print(f"  resampled+fg-cropped shape: {resampled_shape}")

    scale_tf = ScaleIntensityRange(a_min=HU_A_MIN, a_max=HU_A_MAX, b_min=0, b_max=1, clip=True)
    data = scale_tf(data)

    orient_tf = Orientation(axcodes="RAS")
    data = orient_tf(data)

    x = data.unsqueeze(0).to(device=device, dtype=torch.float32)  # [1, 1, X, Y, Z]
    class_vector = torch.tensor(organ_ids, dtype=torch.long, device=device).unsqueeze(-1)  # [B, 1]

    inferer = SlidingWindowInfererAdapt(
        roi_size=PATCH_SIZE, sw_batch_size=1, overlap=overlap,
        with_coord=True, padding_mode="replicate", mode="gaussian",
    )

    with torch.no_grad(), torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
        logits = inferer(
            x, model, transpose=True,
            point_coords=None, point_labels=None,
            class_vector=class_vector, prompt_class=class_vector,
            prev_mask=None, labels=None, label_set=None,
        )  # [1, B, X, Y, Z]

    logits = logits[0].float().cpu()  # [B, X, Y, Z]
    post = VistaPostTransformd(keys=["pred"])
    label_prompt = torch.tensor(organ_ids, dtype=torch.long)
    out = post({"pred": logits, "label_prompt": label_prompt})
    label_map = out["pred"][0].numpy()  # [X, Y, Z], values are organ ids or 0=background

    found_labels = set(int(v) for v in np.unique(label_map) if v != 0)
    fg_mask = label_map != 0

    if not fg_mask.any():
        return {
            "bbox_min_mm": None, "bbox_max_mm": None,
            "found_labels": found_labels, "resampled_shape": resampled_shape,
            "status": "no_foreground_found",
        }

    coords = np.argwhere(fg_mask)  # [N, 3] voxel indices in the RAS, resampled+fg-cropped grid
    vox_min = coords.min(axis=0)
    vox_max = coords.max(axis=0) + 1  # exclusive -> inclusive corner

    affine = np.asarray(data.affine)  # affine of the RAS-oriented, fg-cropped, resampled grid
    corners_vox = np.array([
        [vox_min[0], vox_min[1], vox_min[2], 1],
        [vox_max[0], vox_max[1], vox_max[2], 1],
    ])
    corners_mm = (affine @ corners_vox.T).T[:, :3]
    bbox_min_mm = corners_mm.min(axis=0) - margin_mm
    bbox_max_mm = corners_mm.max(axis=0) + margin_mm

    # weighted centroid (voxel-count-weighted, i.e. just the mean voxel position)
    # of ONLY the core organ set, in physical mm -- used to place a fixed-size
    # z-window rather than the full union bbox, which is dominated by rectum/
    # colon/esophagus at the anatomical extremes
    core_mask = np.isin(label_map, CORE_ORGAN_IDS_FOR_CENTROID)
    core_centroid_mm = None
    if core_mask.any():
        core_coords_vox = np.argwhere(core_mask)
        centroid_vox = core_coords_vox.mean(axis=0)
        centroid_vox_h = np.array([centroid_vox[0], centroid_vox[1], centroid_vox[2], 1])
        centroid_mm = affine @ centroid_vox_h
        core_centroid_mm = tuple(centroid_mm[:3].tolist())

    result = {
        "bbox_min_mm": tuple(bbox_min_mm.tolist()),
        "bbox_max_mm": tuple(bbox_max_mm.tolist()),
        "found_labels": found_labels,
        "resampled_shape": resampled_shape,
        "status": "ok",
        "core_centroid_mm": core_centroid_mm,
    }
    if return_label_map:
        result["label_map"] = label_map
        result["label_map_affine"] = affine
    return result
    return {
    }


if __name__ == "__main__":
    import sys

    fp = sys.argv[1] if len(sys.argv) > 1 else None
    if fp is None:
        print("usage: python roi_localizer.py <path/to/volume.nii.gz>")
        sys.exit(1)
    result = segment_organ_bbox_mm(fp, verbose=True)
    print(result)
