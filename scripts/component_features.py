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


def compute_component_features(component_mask: np.ndarray, body_mask: np.ndarray, spacing) -> dict:
    """component_mask, body_mask: same-shape boolean 3D arrays.
    spacing: (sx, sy, sz) mm per voxel (native, possibly anisotropic)."""
    spacing = np.asarray(spacing, dtype=np.float64)
    voxel_volume_mm3 = float(np.prod(spacing))
    n_voxels = int(component_mask.sum())
    volume_ml = n_voxels * voxel_volume_mm3 / 1000.0

    coords_vox = np.argwhere(component_mask).astype(np.float64)
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
    dilated_bg = ndimage.binary_dilation(~body_mask, structure=struct6)
    wall_contact_voxels = int((component_mask & dilated_bg).sum())
    wall_contact_fraction = wall_contact_voxels / n_voxels if n_voxels > 0 else 0.0

    # centroid-to-wall distance (mm): reuses the same distance-transform
    # approach as qc_air_mask_extended.py's centrality_score, but at the
    # component centroid specifically (one number per component)
    dist_to_outside = ndimage.distance_transform_edt(body_mask, sampling=spacing)
    cv = np.clip(np.round(centroid_vox).astype(int), 0, np.array(body_mask.shape) - 1)
    centroid_wall_distance_mm = float(dist_to_outside[cv[0], cv[1], cv[2]])

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
