"""
Adaptive, per-case ROI crop -- replaces the fixed-mm-offset-from-a-point
scheme (VISTA3D core-organ centroid +/- fixed 100mm/180mm z-margin, see
roi_localizer.py / compute_fixed_z_window.py) with a boundary DERIVED from
each volume's own actual tissue extent, on all 3 axes including z.

Design ported from a teammate's mentor_mil_2 restart (tissue_contour_crop.py
+ tight_4side_crop.py), reimplemented here self-contained (no dependency on
their roi_builder.py / body_mask_bbox_crop.py, which aren't in this repo) so
it can be run and validated against our own GT set directly. Two real
differences from their version:
  1. Their tight_4side_crop.py only extends x/y with the per-ray scan and
     leaves z at the margin-only bound -- their own
     _tissue_start_from_low_edge/_tissue_start_from_high_edge helpers are
     already axis-generic (take an `axis` param), just never invoked for
     axis=2. Here they're applied to z too.
  2. Operates on nibabel-loaded RAS-canonical volumes directly (no
     roi_builder.load_ras dependency).

Rationale for why "adaptive" beats "fixed offset from a point" (organ
centroid, or any single anatomical landmark like femur/vertebra): confirmed
empirically this session (viz_inferior_margin_sweep.py) that even a 20mm
change to the fixed inferior margin left thousands of real GT free-air
voxels still beyond the boundary on a known failure case -- a single
constant cannot track patient-to-patient variation in body extent. Deriving
the boundary from each volume's own measured tissue extent removes that
failure mode by construction.
"""
import numpy as np
from scipy import ndimage

# --- stage 1: tissue-contour body mask (same design as tissue_contour_crop.py) ---
TISSUE_THRESHOLD = -300.0   # well above the CT table's -850..-950 HU signature
DILATION_ITERS = 2          # recovers the skin edge eroded by thresholding
MARGIN_MM = 10.0            # safety margin added to the raw mask extent

# --- stage 2: per-ray tight-extension scan (same design as tight_4side_crop.py) ---
TABLE_BG_HU_MAX = -500.0    # table/background-range HU ceiling for the ray scan


def _keep_largest_2d_per_slice(mask):
    """Arms/hands resting beside the torso can be 3D-connected to it via the
    shoulder at some other z, so a 3D largest-CC can't separate them --
    per-slice 2D largest-component filtering drops any such appendage that
    isn't part of the torso's own cross-section at that particular slice."""
    out = np.zeros_like(mask)
    for z in range(mask.shape[2]):
        sl = mask[:, :, z]
        if not sl.any():
            continue
        labeled2d, n2d = ndimage.label(sl, structure=np.ones((3, 3)))
        if n2d == 0:
            continue
        sizes = ndimage.sum(np.ones_like(labeled2d), labeled2d, index=range(1, n2d + 1))
        largest = int(np.argmax(sizes)) + 1
        out[:, :, z] = labeled2d == largest
    return out


def tissue_contour_mask(hu_array, tissue_threshold=TISSUE_THRESHOLD, dilation_iters=DILATION_ITERS):
    tissue_mask = hu_array > tissue_threshold
    labeled, n = ndimage.label(tissue_mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return np.zeros_like(hu_array, dtype=bool)
    sizes = ndimage.sum(tissue_mask, labeled, range(1, n + 1))
    body_label = int(np.argmax(sizes)) + 1
    body_silhouette = labeled == body_label
    body_mask = ndimage.binary_fill_holes(body_silhouette)
    body_mask = _keep_largest_2d_per_slice(body_mask)
    body_mask = ndimage.binary_fill_holes(body_mask)
    body_mask = ndimage.binary_dilation(body_mask, iterations=dilation_iters)
    return body_mask


def compute_margin_bbox(hu_array, zooms, margin_mm=MARGIN_MM):
    """v3-equivalent: tissue-contour mask extent + a flat margin, all 3 axes."""
    body_mask_full = tissue_contour_mask(hu_array)
    coords = np.where(body_mask_full)
    bbox = []
    info = {}
    for axis in range(3):
        idx_min, idx_max = int(coords[axis].min()), int(coords[axis].max())
        margin_vox = int(np.ceil(margin_mm / zooms[axis]))
        lo = max(0, idx_min - margin_vox)
        hi = min(hu_array.shape[axis], idx_max + margin_vox + 1)
        bbox.append(slice(lo, hi))
        info[f"axis{axis}_body_range"] = (idx_min, idx_max)
    return tuple(bbox), body_mask_full, info


def _tissue_start_from_low_edge(hu_array, axis, other_slices):
    """Restrict to other_slices on the other two axes, walk the FULL range
    on `axis` from index 0, and find the index where the longest low-HU run
    starting at index 0 ends -- i.e. the tightest boundary needed at the LOW
    edge of `axis`. Axis-generic: works identically for x, y, or z."""
    slicer = list(other_slices)
    slicer.insert(axis, slice(None))
    sub = hu_array[tuple(slicer)]
    low = (sub < TABLE_BG_HU_MAX).astype(np.int8)
    cum = np.cumprod(low, axis=axis)
    run = int(cum.sum(axis=axis).max())
    return run


def _tissue_start_from_high_edge(hu_array, axis, other_slices):
    """Same, but from the HIGH edge."""
    slicer = list(other_slices)
    slicer.insert(axis, slice(None))
    sub = hu_array[tuple(slicer)]
    flipped = np.flip(sub, axis=axis)
    low = (flipped < TABLE_BG_HU_MAX).astype(np.int8)
    cum = np.cumprod(low, axis=axis)
    run = int(cum.sum(axis=axis).max())
    axis_len = hu_array.shape[axis]
    return axis_len - run


def compute_tight_3axis_bbox(hu_array, zooms):
    """Full adaptive crop: margin-bbox baseline (all 3 axes), then per-ray
    tight-extension scan applied to ALL 3 axes (x, y, AND z -- the teammate's
    tight_4side_crop.py only did x/y; this is the direct extension to z).
    Only ever extends the margin-bbox outward, never shrinks it -- same
    safety invariant as the original design.
    """
    margin_bbox, mask, info = compute_margin_bbox(hu_array, zooms)
    bounds = [(s.start, s.stop) for s in margin_bbox]
    body_ranges = [info[f"axis{a}_body_range"] for a in range(3)]
    shape = hu_array.shape

    new_bounds = list(bounds)
    for axis in range(3):
        other_axes = [a for a in range(3) if a != axis]
        other_slices = tuple(slice(body_ranges[a][0], body_ranges[a][1] + 1) for a in other_axes)

        tight_lo = _tissue_start_from_low_edge(hu_array, axis, other_slices)
        tight_hi = _tissue_start_from_high_edge(hu_array, axis, other_slices)

        lo0, hi0 = bounds[axis]
        new_lo = min(lo0, tight_lo)
        new_hi = max(hi0, tight_hi)
        new_bounds[axis] = (max(0, new_lo), min(shape[axis], new_hi))

    new_bbox = tuple(slice(lo, hi) for lo, hi in new_bounds)
    diag = {"margin_bbox": bounds, "tight_bbox": new_bounds}
    return new_bbox, margin_bbox, mask, diag


def load_ras_native(filepath):
    """Load a volume canonicalized to RAS, native resolution (no resample) --
    self-contained replacement for roi_builder.load_ras."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(filepath))
    hu_array = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    return hu_array, img.affine, zooms


def apply_crop(hu_array, affine, bbox):
    offset = np.array([s.start for s in bbox] + [0])
    new_affine = affine.copy()
    new_affine[:, 3] = affine @ offset
    return hu_array[bbox], new_affine


# cap applied to the ADAPTIVE z-extent when it's larger than this, so
# extended-FOV volumes (400-600mm+ native z-extent, real tissue throughout)
# don't stay uniformly larger in z than abdomen-only volumes (natural range
# ~250-320mm, see analyze_bbox_consistency.py) -- that size mismatch is
# exactly the resize-compression shortcut-learning signal diagnosed and
# fixed earlier this project (see PROJECT_BRIEFING.md / check_axis_bug.py
# history): a volume with more real anatomy squeezed into the same
# resize-to-patch_size voxel grid is trivially distinguishable as
# "extended-FOV protocol" regardless of pathology. The adaptive crop alone
# fixes clipping (0/23 GT loss, confirmed) but does NOT fix this by itself,
# since real tissue is present at nearly every z-slice in a full-torso scan
# -- so a size cap is layered on top, trimmed from the INFERIOR side first
# (biased toward keeping the superior/diaphragm region, where
# pneumoperitoneum free air actually collects), only when needed.
TARGET_Z_WINDOW_MM = 280.0


def cap_z_extent(bbox, affine, target_mm=TARGET_Z_WINDOW_MM):
    """Given a full 3-axis voxel bbox (from compute_tight_3axis_bbox), trim
    the z bound if its physical extent exceeds target_mm -- cutting from the
    inferior side first, never touching x/y, and never trimming below the
    original (already GT-validated-safe) extent's superior edge.
    """
    z_lo, z_hi = bbox[2].start, bbox[2].stop  # voxel indices, exclusive stop
    z_sign = np.sign(affine[2, 2]) or 1.0  # +1: increasing voxel z = increasing S (superior)
    z_zoom = abs(affine[2, 2])

    extent_mm = (z_hi - z_lo) * z_zoom
    if extent_mm <= target_mm:
        return bbox, {"capped": False, "extent_mm": extent_mm}

    target_vox = int(np.ceil(target_mm / z_zoom))
    if z_sign > 0:
        # increasing index = more superior -> keep the HIGH end (superior),
        # trim from the LOW end (inferior)
        new_z_lo = z_hi - target_vox
        new_bbox = (bbox[0], bbox[1], slice(max(z_lo, new_z_lo), z_hi))
    else:
        # increasing index = more inferior -> keep the LOW end (superior),
        # trim from the HIGH end (inferior)
        new_z_hi = z_lo + target_vox
        new_bbox = (bbox[0], bbox[1], slice(z_lo, min(z_hi, new_z_hi)))

    return new_bbox, {"capped": True, "extent_mm": extent_mm, "target_vox": target_vox}


def compute_final_bbox(hu_array, zooms, affine, target_z_mm=TARGET_Z_WINDOW_MM):
    """Full pipeline: adaptive tight 3-axis bbox (safety: no GT clipping),
    then a z-size cap (protocol-shortcut-learning safety) applied only when
    the adaptive z-extent exceeds target_z_mm."""
    bbox, margin_bbox, mask, diag = compute_tight_3axis_bbox(hu_array, zooms)
    capped_bbox, cap_info = cap_z_extent(bbox, affine, target_mm=target_z_mm)
    diag["z_cap"] = cap_info
    return capped_bbox, bbox, mask, diag


def bbox_vox_to_mm(affine, bbox):
    """Convert a voxel-index bbox (3 slices, on this volume's own native RAS
    grid) to a physical RAS mm bbox -- the format roi_bboxes.csv/dataset.py
    already expect (resample-invariant, so it's reusable against any
    resampled grid of the same volume, same convention as roi_localizer.py's
    segment_organ_bbox_mm output). Takes element-wise min/max of the two
    opposite corners' mapped mm coordinates, not just corner0 vs corner1
    directly, in case the affine has any axis-flip (negative scale) left
    after RAS canonicalization.
    """
    vox_min = np.array([s.start for s in bbox])
    vox_max = np.array([s.stop for s in bbox])  # exclusive stop -> outer corner
    corners_vox = np.array([
        [vox_min[0], vox_min[1], vox_min[2], 1],
        [vox_max[0], vox_max[1], vox_max[2], 1],
    ])
    corners_mm = (affine @ corners_vox.T).T[:, :3]
    bbox_min_mm = corners_mm.min(axis=0)
    bbox_max_mm = corners_mm.max(axis=0)
    return tuple(bbox_min_mm.tolist()), tuple(bbox_max_mm.tolist())
