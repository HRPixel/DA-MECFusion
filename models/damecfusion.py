"""DA-MECFusion V1 main model assembly."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.decoder import FusionDecoder  # noqa: E402
from models.encoders import SimpleEncoder  # noqa: E402
from models.mec_module import MECModule  # noqa: E402
from models.priors import RuleBasedPriorModule, rgb_to_y  # noqa: E402


class DAMECFusionV1(nn.Module):
    """Minimal DA-MECFusion V1 forward pipeline."""

    def __init__(
        self,
        feature_channels: int = 64,
        hidden_channels: int = 64,
        encoder_blocks: int = 2,
        decoder_blocks: int = 2,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

        self.prior_module = RuleBasedPriorModule()
        self.ir_encoder = SimpleEncoder(
            in_channels=1,
            out_channels=feature_channels,
            num_blocks=encoder_blocks,
            base_channels=hidden_channels,
            use_bn=use_bn,
        )
        self.vis_encoder = SimpleEncoder(
            in_channels=1,
            out_channels=feature_channels,
            num_blocks=encoder_blocks,
            base_channels=hidden_channels,
            use_bn=use_bn,
        )
        self.mec = MECModule(
            feature_channels=feature_channels,
            hidden_channels=hidden_channels,
            use_bn=use_bn,
        )
        self.decoder = FusionDecoder(
            in_channels=feature_channels,
            hidden_channels=hidden_channels,
            num_blocks=decoder_blocks,
            out_channels=1,
        )

    def forward(self, ir: torch.Tensor, vis: torch.Tensor) -> dict[str, torch.Tensor]:
        if ir.dim() != 4 or ir.size(1) != 1:
            raise ValueError(f"Expected ir with shape [B,1,H,W], got {tuple(ir.shape)}")
        if vis.dim() != 4 or vis.size(1) != 3:
            raise ValueError(f"Expected vis with shape [B,3,H,W], got {tuple(vis.shape)}")
        if ir.size(0) != vis.size(0) or ir.shape[-2:] != vis.shape[-2:]:
            raise ValueError(
                f"ir and vis must share batch and spatial size, got "
                f"{tuple(ir.shape)} and {tuple(vis.shape)}"
            )

        vis_y = rgb_to_y(vis)
        feat_ir = self.ir_encoder(ir)
        feat_vis = self.vis_encoder(vis_y)

        thermal_prior, sat_uncertainty, smoke_prior = self.prior_module(ir, vis)
        fused_feat, w_ir, w_vis = self.mec(
            feat_ir,
            feat_vis,
            thermal_prior,
            sat_uncertainty,
            smoke_prior,
        )
        fused = self.decoder(fused_feat)

        return {
            "fused": fused,
            "thermal_prior": thermal_prior,
            "sat_uncertainty": sat_uncertainty,
            "smoke_prior": smoke_prior,
            "w_ir": w_ir,
            "w_vis": w_vis,
        }


def _print_output_stats(name: str, x: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(x.shape)}, "
        f"min={x.min().item():.6f}, max={x.max().item():.6f}"
    )


if __name__ == "__main__":
    ir = torch.rand(2, 1, 256, 256)
    vis = torch.rand(2, 3, 256, 256)
    model = DAMECFusionV1(feature_channels=64)

    outputs = model(ir, vis)
    for key, value in outputs.items():
        _print_output_stats(key, value)

    expected_shape = (2, 1, 256, 256)
    for key in ["fused", "thermal_prior", "sat_uncertainty", "smoke_prior", "w_ir", "w_vis"]:
        assert outputs[key].shape == expected_shape, f"{key}: {outputs[key].shape}"

    weight_sum_error = torch.max(torch.abs(outputs["w_ir"] + outputs["w_vis"] - 1.0)).item()
    assert weight_sum_error < 1e-5, weight_sum_error

    loss = outputs["fused"].mean()
    loss.backward()
    print("DAMECFusionV1 self-test passed.")
