"""
Confirms the 2-channel (air-mask) encoder is, before any training, exactly
equivalent to the original 1-channel encoder when the extra channel is zero:
conv_init.weight channel 0 holds the identical pretrained weights in both
cases, and the new channel is zero-initialized, so it contributes nothing to
the first conv's output sum when the input's 2nd channel is all zeros.

This must pass BEFORE trusting any air-mask training run -- if it doesn't,
the "zero-init channel = no-op" premise is wrong and the model isn't starting
from the pretrained encoder's actual learned features.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import VistaClassifier  # noqa: E402


def main():
    torch.manual_seed(0)

    model_1ch = VistaClassifier(freeze_encoder=True, in_channels=1)
    model_2ch = VistaClassifier(freeze_encoder=True, in_channels=2)
    model_1ch.eval()
    model_2ch.eval()

    # copy the 1-channel model's head weights into the 2-channel model so the
    # comparison isolates the ENCODER's behavior -- both models are freshly
    # constructed with random head init otherwise, which would make outputs
    # differ regardless of the encoder being correct
    model_2ch.head.load_state_dict(model_1ch.head.state_dict())

    D = H = W = 96  # smaller than 224 for a fast check; shape doesn't matter for this test
    x1 = torch.rand(1, 1, D, H, W)
    x2 = torch.cat([x1, torch.zeros_like(x1)], dim=1)  # channel 1 = intensity (same values), channel 2 = zeros

    assert model_2ch.encoder.conv_init.weight.shape == (48, 2, 3, 3, 3), model_2ch.encoder.conv_init.weight.shape
    assert torch.equal(model_2ch.encoder.conv_init.weight[:, 0:1], model_1ch.encoder.conv_init.weight), \
        "channel 0 of the 2-channel conv_init does not match the 1-channel pretrained weights"
    assert torch.count_nonzero(model_2ch.encoder.conv_init.weight[:, 1:2]) == 0, \
        "the new (2nd) channel's conv_init weights are not all zero"
    print("[weights] conv_init channel 0 == pretrained 1-channel weights, channel 1 == all zero -- OK")

    with torch.no_grad():
        out1 = model_1ch(x1)
        out2 = model_2ch(x2)

    max_abs_diff = (out1 - out2).abs().max().item()
    print(f"[forward] 1-channel output: {out1.tolist()}")
    print(f"[forward] 2-channel output (mask channel = 0): {out2.tolist()}")
    print(f"[forward] max abs diff: {max_abs_diff:.3e}")

    assert torch.allclose(out1, out2, atol=1e-4), (
        f"2-channel model with a zeroed mask channel does NOT match the 1-channel model "
        f"(max abs diff={max_abs_diff:.3e}) -- the zero-init premise is broken"
    )
    print("\nAll checks passed: with the air-mask channel zeroed, the 2-channel model is "
          "numerically equivalent to the original 1-channel pretrained model.")


if __name__ == "__main__":
    main()
