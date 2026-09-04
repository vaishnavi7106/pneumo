"""
3D Grad-CAM for the fine-tuned VistaClassifier. Primary target: stage 4's
last conv (8^3 spatial, the fine-tuned block -- coarse but most class-
specific since it's the block that was actually adapted to this task).
Secondary: stage 3's last conv (16^3 -- finer localization, less
class-specific since stage 3 stayed frozen).

Key correctness point: the model's forward() normally wraps frozen-stage
computation in torch.no_grad() for efficiency (see model.py). Grad-CAM needs
real gradients flowing to the hooked activation, including for the STAGE 3
hook even though stage 3's own weights are frozen (requires_grad=False).
The fix is standard autograd behavior: setting requires_grad_(True) on the
INPUT tensor makes every downstream op build a graph node regardless of
whether that op's weights require grad (an op requires grad if ANY of its
inputs do) -- so as long as we (a) never enter the no_grad branch and (b)
mark the input as requiring grad, both stage 3 and stage 4 hooks work.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import VistaClassifier  # noqa: E402

TARGET_LAYERS = {
    # hook the whole last SegResBlock, not its inner conv2 -- SegResBlock.forward
    # does `x += identity` (in-place) right after conv2, which corrupts the
    # backward hook's autograd bookkeeping ("view is being modified inplace")
    # if conv2 itself is hooked. The block's own output is never mutated
    # afterward (only passed to the next block/downsample, both out-of-place).
    "stage4": lambda model: model.encoder.layers[4]["blocks"][-1],
    "stage3": lambda model: model.encoder.layers[3]["blocks"][-1],
}


class GradCAM3D:
    def __init__(self, model: VistaClassifier, layer_name: str = "stage4"):
        assert layer_name in TARGET_LAYERS, f"layer_name must be one of {list(TARGET_LAYERS)}"
        self.model = model
        self.layer_name = layer_name
        self.layer = TARGET_LAYERS[layer_name](model)
        self._activation = None
        self._gradient = None
        self._fwd_handle = self.layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = self.layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self._activation = output

    def _backward_hook(self, module, grad_input, grad_output):
        self._gradient = grad_output[0]

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, x: torch.Tensor, target_size=None):
        """
        x: [1, 1, D, H, W] preprocessed input (NOT wrapped in no_grad by caller).
        Returns: (cam [D,H,W] numpy in [0,1] upsampled to target_size, logit float, prob float)
        """
        assert x.shape[0] == 1, "Grad-CAM here assumes batch_size=1"
        assert not self.model.freeze_encoder, (
            "model.freeze_encoder must be False (i.e. set_encoder_finetune_stages() was called) "
            "so forward() does not wrap the encoder call in torch.no_grad() -- otherwise no "
            "gradient can reach either hooked layer regardless of the input's requires_grad."
        )

        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)  # forces autograd tracking through frozen-weight layers too

        logit = self.model(x)  # [1]
        prob = torch.sigmoid(logit).item()

        logit.backward()

        activation = self._activation[0]  # [C, d, h, w]
        gradient = self._gradient[0]      # [C, d, h, w]

        assert activation is not None and gradient is not None, (
            f"no activation/gradient captured for {self.layer_name} -- hook did not fire "
            f"or backward did not reach this layer"
        )
        assert not torch.isnan(gradient).any() and not torch.isinf(gradient).any(), (
            f"NaN/Inf in gradient at {self.layer_name}"
        )

        weights = gradient.mean(dim=(1, 2, 3))  # [C] global-average-pooled gradient per channel
        cam = (weights[:, None, None, None] * activation).sum(dim=0)  # [d, h, w]
        cam = F.relu(cam)

        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max
        else:
            # all-zero CAM (ReLU killed everything) -- degenerate but not an error;
            # caller should treat this case as "no localized signal" rather than crash
            pass

        if target_size is not None:
            cam_up = F.interpolate(cam[None, None], size=target_size, mode="trilinear",
                                    align_corners=False)[0, 0]
        else:
            cam_up = cam

        return cam_up.detach().cpu().numpy(), logit.item(), prob


def load_finetuned_model_for_gradcam(checkpoint_path: str, unfreeze_stages=(4,)):
    """Load a fine-tune checkpoint in the SAME partially-unfrozen configuration
    it was trained with, so forward() takes the gradient-enabled code path."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model = VistaClassifier(freeze_encoder=True, hidden_dim=config.get("hidden_dim", 128),
                             dropout=config.get("dropout", 0.3))
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.set_encoder_finetune_stages(list(unfreeze_stages))
    model.to(device)
    model.eval()  # eval mode for BN/dropout -- Grad-CAM still gets real gradients via requires_grad on input
    return model, ckpt
