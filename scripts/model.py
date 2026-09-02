"""
Step 6: extract VISTA3D's pretrained image-encoder backbone (SegResEncoder,
the `image_encoder.encoder.*` weights -- 175M params, 80.5% of image_encoder)
and attach a linear-probe classification head on top of its deepest feature
stage (channels=768 at input 128^3, blocks_down=(1,2,2,4,4), init_filters=48).

The two decoder branches (`image_encoder.up_layers*`) and the segmentation
prompt/class heads (`point_head`, `class_head`) are intentionally dropped --
they exist only for VISTA3D's interactive/auto segmentation task, not for
whole-volume classification.
"""
import os

import torch
import torch.nn as nn
from monai.networks.nets.segresnet_ds import SegResEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_CKPT = os.path.join(ROOT, "vista3d", "models", "model.pt")

# exact VISTA3D SegResEncoder config (from monai.networks.nets.vista3d.vista3d132)
ENCODER_EMBED_DIM = 48
BLOCKS_DOWN = (1, 2, 2, 4, 4)
DEEPEST_CHANNELS = ENCODER_EMBED_DIM * (2 ** (len(BLOCKS_DOWN) - 1))  # 48 * 16 = 768

# exact VISTA3D intensity window (configs/inference.json ScaleIntensityRanged)
HU_A_MIN = -963.8247715525971
HU_A_MAX = 1053.678477684517
RESAMPLE_SPACING = (1.5, 1.5, 1.5)
PATCH_SIZE = (128, 128, 128)


def build_pretrained_encoder(checkpoint_path: str = BUNDLE_CKPT) -> SegResEncoder:
    """Build the SegResEncoder backbone and load VISTA3D's pretrained weights into it."""
    encoder = SegResEncoder(
        spatial_dims=3,
        init_filters=ENCODER_EMBED_DIM,
        in_channels=1,
        norm="instance",
        blocks_down=BLOCKS_DOWN,
    )

    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    prefix = "image_encoder.encoder."
    encoder_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)

    return encoder


class VistaClassifier(nn.Module):
    """Frozen (or fine-tunable) VISTA3D encoder + global-pool linear-probe head."""

    def __init__(self, checkpoint_path: str = BUNDLE_CKPT, freeze_encoder: bool = True,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = build_pretrained_encoder(checkpoint_path)
        self.freeze_encoder = freeze_encoder
        self.set_encoder_trainable(not freeze_encoder)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(DEEPEST_CHANNELS, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def set_encoder_trainable(self, trainable: bool):
        for p in self.encoder.parameters():
            p.requires_grad = trainable
        self.encoder.train(trainable)
        self.finetune_stage_indices = None

    def set_encoder_finetune_stages(self, stage_indices):
        """Partial fine-tuning: freeze the whole encoder except the given
        `encoder.layers[i]` stage indices (0=shallowest .. 4=deepest, 768ch).
        Everything else (conv_init + non-selected stages) stays frozen.

        Sets freeze_encoder=False so forward() stops wrapping the whole
        encoder call in torch.no_grad() -- autograd still automatically skips
        building a graph for the frozen prefix stages (their params AND their
        inputs have requires_grad=False, so no node is created), and only
        starts tracking gradients once it reaches the first unfrozen stage.
        This is the standard "freeze early layers" pattern and needs no
        manual no_grad juggling per-layer.
        """
        n_stages = len(self.encoder.layers)
        assert all(0 <= i < n_stages for i in stage_indices), f"stage indices must be in [0, {n_stages})"
        self.freeze_encoder = False
        self.finetune_stage_indices = sorted(set(stage_indices))

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        for idx in self.finetune_stage_indices:
            for p in self.encoder.layers[idx].parameters():
                p.requires_grad = True

    def train(self, mode: bool = True):
        """Override so a plain model.train() call (standard at the top of every
        training epoch) cannot silently flip frozen encoder stages back to
        train mode -- nn.Module.train() recurses into ALL submodules by default.
        Only genuinely-trainable stages (full unfreeze, or the selected
        fine-tune stages) actually flip to train mode."""
        super().train(mode)
        finetune_stages = getattr(self, "finetune_stage_indices", None)
        if self.freeze_encoder:
            self.encoder.eval()
        elif finetune_stages is not None:
            self.encoder.eval()  # frozen prefix stages (and their norm layers) stay in eval
            for idx in finetune_stages:
                self.encoder.layers[idx].train(mode)
        # else: freeze_encoder=False and no finetune_stage_indices means the
        # WHOLE encoder is trainable (full fine-tune) -- super().train(mode)
        # already handled that correctly, nothing more to do.
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, D, H, W] in [0, 1], already resampled/cropped/oriented
        if self.freeze_encoder:
            # explicit no_grad: encoder activations must NOT be retained for backward
            # since encoder params are frozen -- without this, memory would include
            # a full backward graph through 175M frozen params for no benefit.
            with torch.no_grad():
                stages = self.encoder(x)  # list of 5 multi-scale feature maps
        else:
            stages = self.encoder(x)
        deepest = stages[-1]      # [B, 768, D/16, H/16, W/16]
        pooled = self.pool(deepest)
        logit = self.head(pooled)  # [B, 1]
        return logit.squeeze(-1)


if __name__ == "__main__":
    encoder = build_pretrained_encoder()
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Loaded SegResEncoder backbone: {n_params:,} params ({n_params*4/1e6:.1f} MB fp32)")

    model = VistaClassifier(freeze_encoder=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"VistaClassifier: {total:,} total params, {trainable:,} trainable (linear probe)")

    x = torch.rand(1, 1, *PATCH_SIZE)
    with torch.no_grad():
        stages = model.encoder(x)
        for i, s in enumerate(stages):
            print(f"stage {i}: shape={tuple(s.shape)}")
        out = model(x)
    print(f"classifier output shape: {tuple(out.shape)} (expect (1,))")
