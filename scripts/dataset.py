"""
Step 7/8: PyTorch Dataset + DataLoader implementing the final preprocessing
order for both FOV groups:

  reorient (RAS) -> resample to 1.5mm isotropic -> crop to ROI bbox in
  physical mm space (extended-FOV volumes only; abdomen-only volumes pass
  through unchanged) -> scale HU to [0,1] with VISTA3D's exact window ->
  resize/pad to 128^3

ROI bboxes come from the cached scripts/roi_bboxes.csv (produced by
compute_roi_bboxes.py) -- computing them on the fly per epoch would mean
re-running full VISTA3D segmentation (~9s/volume) on every access, which is
wasteful since the bbox doesn't change.
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.transforms import Orientation, Spacing
from torch.utils.data import DataLoader, Dataset

from augment import augment_volume

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_CSV = os.path.join(ROOT, "scripts", "audit_results.csv")
ROI_BBOX_CSV = os.path.join(ROOT, "scripts", "roi_bboxes.csv")

HU_A_MIN = -963.8247715525971
HU_A_MAX = 1053.678477684517
RESAMPLE_SPACING = (1.5, 1.5, 1.5)
PATCH_SIZE = (128, 128, 128)
EXTENDED_FOV_THRESHOLD_MM = 375.0
BATCH_SIZE = 4

# optional second input channel: a binary/soft air mask from thresholding raw
# HU (computed AFTER resample+crop but BEFORE the [0,1] intensity
# normalization -- see PneumoDataset._preprocess). -600 HU is the suggested
# starting point (air is roughly -1000 to -600 HU); kept as a tunable
# threshold, not hardcoded into the training scripts, for later sweeping.
DEFAULT_AIR_MASK_THRESHOLD = -600.0

# a naive <-600 HU threshold also flags the background air OUTSIDE the
# patient's body (visible in QC as a large contiguous region extending past
# the body silhouette) -- restrict the air mask to inside the body first.
# -500 HU roughly separates body tissue from surrounding background air.
BODY_MASK_HU_THRESHOLD = -500.0

# binary closing applied to the body-tissue threshold BEFORE labeling/
# fill_holes. WITHOUT this, real intra-abdominal gas sitting close to the
# abdominal wall can leak through a thin/blurred (resampling partial-volume)
# low-HU path to the background exterior -- once connected to the border,
# binary_fill_holes correctly does NOT fill it (it's no longer an enclosed
# hole), silently deleting real gas along with the background. Diagnosed and
# confirmed via diagnose_air_mask_fix.py: recovered ~1.2-1.4% of the original
# unrestricted mask (real gas pockets, visually confirmed) without
# reintroducing the background halo. 3 iterations = ~3 voxels = ~4.5mm at the
# 1.5mm isotropic resample spacing.
BODY_MASK_CLOSING_ITERATIONS = 3


def _load_reoriented_resampled(filepath: str) -> MetaTensor:
    img = nib.load(filepath)
    data = img.get_fdata(dtype=np.float32)[None]
    affine = torch.as_tensor(img.affine, dtype=torch.float64)
    data = MetaTensor(torch.as_tensor(data, dtype=torch.float32), affine=affine)
    data = Orientation(axcodes="RAS")(data)
    data = Spacing(pixdim=RESAMPLE_SPACING, mode="bilinear")(data)
    return data


def _crop_to_bbox_mm(data: MetaTensor, bbox_min_mm, bbox_max_mm) -> MetaTensor:
    """Crop a RAS-oriented MetaTensor to a physical-mm bounding box."""
    affine = np.asarray(data.affine)
    inv = np.linalg.inv(affine)
    corners_mm = np.array([
        [bbox_min_mm[0], bbox_min_mm[1], bbox_min_mm[2], 1],
        [bbox_max_mm[0], bbox_max_mm[1], bbox_max_mm[2], 1],
    ])
    corners_vox = (inv @ corners_mm.T).T[:, :3]
    vox_min = np.floor(corners_vox.min(axis=0)).astype(int)
    vox_max = np.ceil(corners_vox.max(axis=0)).astype(int)

    shape = np.array(data.shape[1:])
    vox_min = np.clip(vox_min, 0, shape - 1)
    vox_max = np.clip(vox_max, vox_min + 1, shape)

    return data[:, vox_min[0]:vox_max[0], vox_min[1]:vox_max[1], vox_min[2]:vox_max[2]]


def _compute_body_mask(raw_hu_3d: torch.Tensor, body_threshold: float = BODY_MASK_HU_THRESHOLD) -> torch.Tensor:
    """raw_hu_3d: [X, Y, Z] raw HU. Returns a [X, Y, Z] float mask, 1.0 inside
    the patient's body, 0.0 outside -- so a downstream air threshold can be
    intersected with it to exclude background scanner air.

    body tissue is > body_threshold HU; background air is far below it. Taking
    only the LARGEST connected component of the >threshold region drops any
    small disconnected bright artifacts (table, wires) outside the body, and
    filling holes afterward keeps bowel-gas pockets INSIDE that silhouette
    from being carved out (they're < body_threshold themselves, so without
    hole-filling they'd incorrectly count as "outside the body").
    """
    from scipy import ndimage

    arr = raw_hu_3d.numpy()
    body_binary = arr > body_threshold

    # close BEFORE labeling -- bridges thin leak paths (e.g. a blurred/thin
    # abdominal wall right next to free air) that would otherwise connect an
    # internal gas pocket to the background exterior, causing fill_holes to
    # skip it (see BODY_MASK_CLOSING_ITERATIONS comment above)
    struct = ndimage.generate_binary_structure(3, 1)
    body_binary = ndimage.binary_closing(body_binary, structure=struct,
                                          iterations=BODY_MASK_CLOSING_ITERATIONS)

    labeled, num_components = ndimage.label(body_binary)
    if num_components == 0:
        # degenerate (e.g. an all-air patch) -- no restriction possible, fall
        # back to "everywhere" so the caller's intersection is a no-op
        return torch.ones_like(raw_hu_3d, dtype=torch.float32)

    sizes = ndimage.sum(body_binary, labeled, index=range(1, num_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    body_mask = labeled == largest_label
    body_mask = ndimage.binary_fill_holes(body_mask)

    # erode back by 1 iteration: the gas-pocket-preserving benefit came from
    # fill_holes operating on the CLOSED mask (bridging the leak path so the
    # hole gets recognized at all), not from the final mask itself staying
    # dilated -- eroding back removes the closing's slight boundary dilation
    # (which was letting a thin skin-surface sliver through) while keeping
    # the already-filled interior gas pockets intact
    body_mask = ndimage.binary_erosion(body_mask, structure=struct, iterations=1)

    return torch.as_tensor(body_mask, dtype=torch.float32)


CACHE_DIR = os.path.join(ROOT, "cache")


class PneumoDataset(Dataset):
    def __init__(self, filepaths, audit_csv=AUDIT_CSV, roi_bbox_csv=ROI_BBOX_CSV, patch_size=PATCH_SIZE,
                 cache_dir=CACHE_DIR, use_cache=True, augment=False, air_mask_threshold=None):
        self.filepaths = list(filepaths)
        self.patch_size = tuple(patch_size)
        self.use_cache = use_cache
        # train-only stochastic augmentation, applied AFTER cache load (see
        # augment.py) -- never set True for val/test, or metrics would be
        # measured against a moving target instead of a fixed evaluation set
        self.augment = augment
        # None = single-channel (CT intensity only, original behavior). A float
        # enables a 2nd channel: 1.0 where raw HU < threshold, else 0.0 (soft
        # after resize, since resize uses trilinear interpolation) -- see
        # model.py's build_pretrained_encoder(in_channels=2) for how the
        # encoder's first conv is adapted to consume it.
        self.air_mask_threshold = air_mask_threshold
        # cache is keyed by patch_size AND channel config since the same volume
        # produces a different tensor at 128^3 vs 224^3, and a 1-channel vs
        # 2-channel (air-mask) tensor is a completely different shape/content
        channel_suffix = f"_airmask{air_mask_threshold:g}" if air_mask_threshold is not None else ""
        self.cache_dir = os.path.join(cache_dir, f"{self.patch_size[0]}{channel_suffix}") if use_cache else None
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        audit = pd.read_csv(audit_csv)
        self.meta = audit.set_index("filepath").to_dict(orient="index")

        self.roi = {}
        if os.path.exists(roi_bbox_csv):
            roi_df = pd.read_csv(roi_bbox_csv)
            for _, row in roi_df.iterrows():
                if row.get("status") == "ok":
                    self.roi[row["filepath"]] = (
                        (row["bbox_min_x"], row["bbox_min_y"], row["bbox_min_z"]),
                        (row["bbox_max_x"], row["bbox_max_y"], row["bbox_max_z"]),
                    )

        missing_roi = [
            fp for fp in self.filepaths
            if self.meta[fp]["extent_z_mm"] >= EXTENDED_FOV_THRESHOLD_MM and fp not in self.roi
        ]
        if missing_roi:
            raise ValueError(
                f"{len(missing_roi)} extended-FOV volumes have no cached ROI bbox in "
                f"{roi_bbox_csv} -- run compute_roi_bboxes.py first. First few: {missing_roi[:3]}"
            )

    def __len__(self):
        return len(self.filepaths)

    def _cache_path(self, filepath: str) -> str:
        # hash the full path so cache filenames are filesystem-safe and collision-free
        import hashlib
        key = hashlib.sha1(filepath.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.pt")

    def _preprocess(self, filepath: str) -> torch.Tensor:
        row = self.meta[filepath]
        data = _load_reoriented_resampled(filepath)

        if row["extent_z_mm"] >= EXTENDED_FOV_THRESHOLD_MM:
            bbox_min_mm, bbox_max_mm = self.roi[filepath]
            data = _crop_to_bbox_mm(data, bbox_min_mm, bbox_max_mm)

        # raw HU, post-resample+crop, PRE-normalization -- the air mask (if
        # enabled) is thresholded on this, not on the [0,1]-scaled intensity.
        # shape is [1, X, Y, Z] (leading 1 = channel dim from nib.get_fdata()[None])
        raw_hu = torch.as_tensor(np.asarray(data), dtype=torch.float32)

        x = torch.clamp(raw_hu, HU_A_MIN, HU_A_MAX)
        x = (x - HU_A_MIN) / (HU_A_MAX - HU_A_MIN)
        x = x.unsqueeze(0)  # [1, 1, X, Y, Z] (batch, channel, spatial)
        x = F.interpolate(x, size=self.patch_size, mode="trilinear", align_corners=False)
        x = x.squeeze(0)  # [1, *patch_size]

        if self.air_mask_threshold is not None:
            raw_hu_3d = raw_hu.squeeze(0)  # [X, Y, Z]
            body_mask_3d = _compute_body_mask(raw_hu_3d)  # 1.0 inside the body, 0.0 outside
            # gas only counts if it's INSIDE the body -- without this, background
            # scanner air (also < threshold) gets flagged too, see qc_air_mask.py
            air_binary = (raw_hu_3d < self.air_mask_threshold) & (body_mask_3d > 0.5)

            mask = air_binary.float().unsqueeze(0).unsqueeze(0)  # [1, 1, X, Y, Z]
            # trilinear (not nearest) so the mask resizes with the same
            # interpolation as the intensity channel -- produces a "soft" mask
            # at patch_size resolution (values between 0/1 at former edges)
            mask = F.interpolate(mask, size=self.patch_size, mode="trilinear", align_corners=False)
            mask = mask.squeeze(0)  # [1, *patch_size]
            x = torch.cat([x, mask], dim=0)  # [2, *patch_size]

        return x

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        row = self.meta[filepath]
        label = 1.0 if row["label"] == "positive" else 0.0

        if self.use_cache:
            cache_path = self._cache_path(filepath)
            if os.path.exists(cache_path):
                x = torch.load(cache_path, weights_only=True)
            else:
                x = self._preprocess(filepath)
                # write to a temp file then rename -- atomic, avoids a corrupt
                # cache entry if two workers race or a run is interrupted mid-write
                tmp_path = cache_path + f".tmp{os.getpid()}"
                torch.save(x, tmp_path)
                os.replace(tmp_path, cache_path)
        else:
            x = self._preprocess(filepath)

        if self.augment:
            x = augment_volume(x)

        return x, torch.tensor(label, dtype=torch.float32), filepath


def make_dataloaders(splits, batch_size: int = BATCH_SIZE, num_workers: int = 0, patch_size=PATCH_SIZE,
                      augment_train: bool = False, air_mask_threshold=None):
    loaders = {}
    for name, filepaths in splits.items():
        ds = PneumoDataset(filepaths, patch_size=patch_size, augment=(augment_train and name == "train"),
                            air_mask_threshold=air_mask_threshold)
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"),
            num_workers=num_workers, pin_memory=True,
        )
    return loaders


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from split import patient_level_split  # noqa: E402

    splits, _ = patient_level_split()
    print({k: len(v) for k, v in splits.items()})

    loaders = make_dataloaders(splits, batch_size=BATCH_SIZE)
    train_batch = next(iter(loaders["train"]))
    x, y, paths = train_batch
    print(f"batch x shape: {tuple(x.shape)}, y: {y.tolist()}")
    print(f"x range: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"paths: {[os.path.basename(p) for p in paths]}")
