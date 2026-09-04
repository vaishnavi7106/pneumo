"""
Lightweight 3D augmentation applied ONLY to the train split (dataset.py wires
this in via PneumoDataset(..., augment=True) for "train" and augment=False
for "val"/"test" -- see verify_augmentation_split.py for the check that this
wiring actually holds).

Applied AFTER cache load / preprocessing (crop + resize to patch_size already
done), so caching of the expensive deterministic steps is unaffected -- these
transforms are cheap and re-rolled fresh every __getitem__ call.

Rotation uses padding_mode="border" (edge replication), not zeros, specifically
because zero-padding at a rotated volume's corners can look like a sharp
anatomical edge to a 3D CNN and get learned as a shortcut -- the same class of
risk flagged for the ROI-crop bug earlier this project. Angles are kept small
(+/-10 deg) and rotation is about a single random axis at a time (not a
compound 3-axis rotation) to limit how much real content gets rotated out of
frame, especially given the fixed 280mm z-crop window leaves little margin at
the superior/inferior edges for extended-FOV volumes.
"""
import numpy as np
import torch
import torch.nn.functional as F

FLIP_PROB = 0.5
ROTATE_PROB = 0.5
MAX_ROTATE_DEG = 10.0
INTENSITY_SCALE_RANGE = (0.9, 1.1)
INTENSITY_SHIFT_RANGE = (-0.05, 0.05)


def _rotation_matrix_3x3(axis: int, angle_rad: float) -> torch.Tensor:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 0:  # rotate in the Y-Z plane (about X)
        m = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == 1:  # rotate in the X-Z plane (about Y)
        m = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:  # axis == 2, rotate in the X-Y plane (about Z)
        m = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return torch.tensor(m, dtype=torch.float32)


def random_rotate(x: torch.Tensor, axis: int, angle_deg: float) -> torch.Tensor:
    """x: [1, D, H, W]. Rotates about one of the three volume axes."""
    angle_rad = np.deg2rad(angle_deg)
    rot = _rotation_matrix_3x3(axis, angle_rad)
    theta = torch.zeros(1, 3, 4, dtype=torch.float32)
    theta[0, :, :3] = rot
    grid = F.affine_grid(theta, [1, 1, *x.shape[1:]], align_corners=False)
    x_rot = F.grid_sample(x.unsqueeze(0), grid, mode="bilinear", padding_mode="border",
                           align_corners=False)
    return x_rot.squeeze(0)


def augment_volume(x: torch.Tensor, rng: np.random.RandomState = None) -> torch.Tensor:
    """x: [1, D, H, W] float tensor in [0, 1] (post crop/resize). Returns an
    augmented copy of the same shape, still clamped to [0, 1]."""
    rng = rng or np.random.RandomState()

    # L-R flip only (array axis 1 = dim after channel, corresponds to the R-L
    # physical axis under the RAS orientation convention used throughout this
    # pipeline) -- flipping S-I or A-P would put anatomy in physiologically
    # implausible positions (e.g. head where feet are), L-R is the standard
    # laterality augmentation for abdominal CT since pathology isn't lateralized.
    if rng.random() < FLIP_PROB:
        x = torch.flip(x, dims=[1])

    if rng.random() < ROTATE_PROB:
        axis = int(rng.randint(0, 3))
        angle = rng.uniform(-MAX_ROTATE_DEG, MAX_ROTATE_DEG)
        x = random_rotate(x, axis, angle)

    scale = rng.uniform(*INTENSITY_SCALE_RANGE)
    shift = rng.uniform(*INTENSITY_SHIFT_RANGE)
    x = x * scale + shift
    x = torch.clamp(x, 0.0, 1.0)

    return x
