"""
Per-connected-component feature extraction for the air-mask channel -- stage
2 of a candidate-generation + false-positive-reduction cascade (following the
referenced pneumoperitoneum AI paper's two-stage design, scaled down for 23
GT volumes instead of thousands).

Works in NATIVE (unresampled) voxel space, same as eval_air_mask_vs_gt.py, so
component masks align exactly with the real GT segmentations at merged/GT/ --
no resampling/interpolation error between the two.
"""
import numpy as np
from scipy import ndimage

MIN_COMPONENT_VOXELS = 5


def extract_components(air_mask: np.ndarray, min_voxels: int = MIN_COMPONENT_VOXELS):
    """26-connectivity labeling (permissive -- keeps a genuinely contiguous
    gas collection as one component even where it narrows to a diagonal-only
    connection, rather than spuriously fragmenting it into many pieces).
    Returns (labeled_array, list of component ids kept after the min-size
    filter -- NOT dropped for being small per se, just for being pure noise
    at a few voxels)."""
    struct = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    labeled, num = ndimage.label(air_mask, structure=struct)
    if num == 0:
        return labeled, []
    sizes = ndimage.sum(air_mask, labeled, index=range(1, num + 1))
    kept_ids = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
    return labeled, kept_ids


def crop_to_component_bbox(full_shape, bbox_slice, crop_pad: int = 1):
    """Pad a component's bounding box (e.g. from ndimage.find_objects) by
    crop_pad voxels (needed for 6-connectivity dilation correctness at the
    boundary) and clip to the volume's bounds. Returns (padded_slice,
    global_offset) -- global_offset is padded_slice's [0,0,0] in full-volume
    voxel coordinates, needed to translate local coords back to global."""
    padded_slice = tuple(
        slice(max(0, s.start - crop_pad), min(full_shape[ax], s.stop + crop_pad))
        for ax, s in enumerate(bbox_slice)
    )
    global_offset = np.array([s.start for s in padded_slice], dtype=np.float64)
    return padded_slice, global_offset


def compute_component_features(component_mask_local: np.ndarray, body_mask_local: np.ndarray, spacing,
                                dist_to_outside_global: np.ndarray, global_offset=(0.0, 0.0, 0.0)) -> dict:
    """component_mask_local, body_mask_local: boolean 3D arrays ALREADY
    CROPPED by the caller to (a padded version of) the component's bounding
    box -- see crop_to_component_bbox(). Operating on small local arrays
    instead of the full native volume (~512x512x72) is what makes this fast;
    doing the equivalent full-array ops per component (argwhere,
    binary_dilation, plus the caller's own `labeled == cid` mask extraction)
    was measured at ~140-200ms/component, making a ~150-component volume
    take 20-30+ seconds. Cropped, the whole volume's components finish in a
    fraction of a second.

    spacing: (sx, sy, sz) mm per voxel (native, possibly anisotropic).
    dist_to_outside_global: precomputed ndimage.distance_transform_edt(body_mask,
    sampling=spacing) for the WHOLE volume -- shared across all components
    from the same volume (expensive, identical every time). Only a single
    point is indexed from it here, translated via global_offset, so it does
    NOT need cropping itself.
    global_offset: padded_slice's [0,0,0] in full-volume voxel coordinates
    (from crop_to_component_bbox), used to translate local centroid ->
    global coordinates for the dist_to_outside_global lookup.
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    global_offset = np.asarray(global_offset, dtype=np.float64)
    voxel_volume_mm3 = float(np.prod(spacing))

    comp_local = component_mask_local
    body_local = body_mask_local

    n_voxels = int(comp_local.sum())
    volume_ml = n_voxels * voxel_volume_mm3 / 1000.0

    coords_vox_local = np.argwhere(comp_local).astype(np.float64)
    coords_vox = coords_vox_local + global_offset[None, :]  # back to full-volume voxel coords
    coords_mm = coords_vox * spacing[None, :]
    centroid_mm = coords_mm.mean(axis=0)
    centroid_vox = coords_vox.mean(axis=0)

    # elongation: ratio of largest to smallest principal-axis extent, in
    # physical mm (PCA on mm-scaled coordinates so anisotropic voxels don't
    # distort the shape) -- a thin wall-hugging free-air layer should score
    # high here; a round bowel-gas bubble should score near 1.
    if len(coords_mm) >= 3:
        centered = coords_mm - centroid_mm
        cov = np.cov(centered.T)
        eigvals = np.clip(np.linalg.eigvalsh(cov), 1e-9, None)
        # a component only 1 voxel thick along one axis has a near-zero
        # eigenvalue there, blowing the ratio up to ~1e5+ (verified via a
        # synthetic flat-sheet test) -- capped since beyond this the exact
        # value is meaningless noise that would dominate feature scaling
        elongation = float(min(np.sqrt(eigvals.max() / eigvals.min()), 50.0))
    else:
        elongation = 1.0

    # wall-contact fraction: fraction of component voxels with >=1 face-
    # neighbor outside the body (body_mask == 0) -- proxies "hugging the
    # abdominal wall" vs. sitting deep in the interior (e.g. gastric bubble)
    struct6 = ndimage.generate_binary_structure(3, 1)
    dilated_bg_local = ndimage.binary_dilation(~body_local, structure=struct6)
    wall_contact_voxels = int((comp_local & dilated_bg_local).sum())
    wall_contact_fraction = wall_contact_voxels / n_voxels if n_voxels > 0 else 0.0

    # centroid-to-wall distance (mm): reuses the same distance-transform
    # approach as qc_air_mask_extended.py's centrality_score, but at the
    # component centroid specifically (one number per component)
    cv = np.clip(np.round(centroid_vox).astype(int), 0, np.array(dist_to_outside_global.shape) - 1)
    centroid_wall_distance_mm = float(dist_to_outside_global[cv[0], cv[1], cv[2]])

    # compactness: volume / bounding-box volume, in physical mm^3 -- a
    # simple proxy for sphericity that avoids needing marching-cubes surface
    # extraction. 1.0 = fills its bounding box (blocky/round); low = thin or
    # irregular (a wall-hugging free-air sheet is thin relative to its
    # bounding box and should score low here).
    bbox_vox = coords_vox.max(axis=0) - coords_vox.min(axis=0) + 1
    bbox_mm3 = float(np.prod(bbox_vox * spacing))
    compactness = (volume_ml * 1000.0 / bbox_mm3) if bbox_mm3 > 0 else 0.0

    return {
        "n_voxels": n_voxels,
        "volume_ml": volume_ml,
        "elongation": elongation,
        "wall_contact_fraction": wall_contact_fraction,
        "centroid_wall_distance_mm": centroid_wall_distance_mm,
        "compactness": compactness,
    }
