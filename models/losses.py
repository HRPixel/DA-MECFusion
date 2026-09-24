"""Loss functions for DA-MECFusion V1 and V2-Base."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.priors import rgb_to_y, sobel_components, sobel_gradient  # noqa: E402


class DAMECFusionLoss(nn.Module):
    """Fusion loss with configurable intensity and gradient targets."""

    def __init__(
        self,
        lambda_int: float = 1.0,
        lambda_grad: float = 10.0,
        lambda_thermal: float = 1.0,
        lambda_sat: float = 0.5,
        gradient_mode: str = "magnitude",
        intensity_target: str = "max",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if gradient_mode not in {"magnitude", "directional"}:
            raise ValueError("gradient_mode must be 'magnitude' or 'directional'")
        if intensity_target not in {"max", "mec_weighted"}:
            raise ValueError("intensity_target must be 'max' or 'mec_weighted'")
        self.lambda_int = lambda_int
        self.lambda_grad = lambda_grad
        self.lambda_thermal = lambda_thermal
        self.lambda_sat = lambda_sat
        self.gradient_mode = gradient_mode
        self.intensity_target = intensity_target
        self.eps = eps

    @staticmethod
    def _require_key(outputs: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in outputs:
            raise KeyError(f"outputs must contain '{key}'")
        return outputs[key]

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        ir: torch.Tensor,
        vis: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fused = self._require_key(outputs, "fused")
        thermal_prior = self._require_key(outputs, "thermal_prior")
        sat_uncertainty = self._require_key(outputs, "sat_uncertainty")
        self._require_key(outputs, "smoke_prior")
        w_ir = self._require_key(outputs, "w_ir")
        w_vis = self._require_key(outputs, "w_vis")

        if ir.dim() != 4 or ir.size(1) != 1:
            raise ValueError(f"Expected ir with shape [B,1,H,W], got {tuple(ir.shape)}")
        if vis.dim() != 4 or vis.size(1) != 3:
            raise ValueError(f"Expected vis with shape [B,3,H,W], got {tuple(vis.shape)}")
        if fused.shape != ir.shape:
            raise ValueError(f"fused and ir must share shape, got {tuple(fused.shape)} and {tuple(ir.shape)}")
        if thermal_prior.shape != ir.shape:
            raise ValueError(
                f"thermal_prior and ir must share shape, got {tuple(thermal_prior.shape)} and {tuple(ir.shape)}"
            )
        if sat_uncertainty.shape != ir.shape:
            raise ValueError(
                f"sat_uncertainty and ir must share shape, got {tuple(sat_uncertainty.shape)} and {tuple(ir.shape)}"
            )

        vis_y = rgb_to_y(vis)

        if self.intensity_target == "max":
            target_int = torch.maximum(ir, vis_y)
        else:
            if w_ir.shape != ir.shape or w_vis.shape != ir.shape:
                raise ValueError(
                    "mec_weighted intensity target requires w_ir and w_vis "
                    f"to match IR shape, got {tuple(w_ir.shape)}, {tuple(w_vis.shape)}, "
                    f"and {tuple(ir.shape)}"
                )
            target_int = w_ir.detach() * ir + w_vis.detach() * vis_y
        loss_int = F.l1_loss(fused, target_int)

        if self.gradient_mode == "magnitude":
            grad_f = sobel_gradient(fused, eps=self.eps)
            grad_ir = sobel_gradient(ir, eps=self.eps)
            grad_vis = sobel_gradient(vis_y, eps=self.eps)
            loss_grad = F.l1_loss(grad_f, torch.maximum(grad_ir, grad_vis))
        else:
            gx_f, gy_f = sobel_components(fused)
            gx_ir, gy_ir = sobel_components(ir)
            gx_vis, gy_vis = sobel_components(vis_y)
            mag_ir = torch.sqrt(gx_ir * gx_ir + gy_ir * gy_ir + self.eps)
            mag_vis = torch.sqrt(gx_vis * gx_vis + gy_vis * gy_vis + self.eps)
            use_ir = mag_ir >= mag_vis
            target_gx = torch.where(use_ir, gx_ir, gx_vis)
            target_gy = torch.where(use_ir, gy_ir, gy_vis)
            loss_grad = F.l1_loss(gx_f, target_gx) + F.l1_loss(gy_f, target_gy)

        loss_thermal = torch.mean(torch.abs(thermal_prior * (fused - ir)))
        loss_sat = torch.mean(torch.abs(sat_uncertainty * (fused - ir)))

        loss_total = (
            self.lambda_int * loss_int
            + self.lambda_grad * loss_grad
            + self.lambda_thermal * loss_thermal
            + self.lambda_sat * loss_sat
        )

        loss_dict = {
            "loss_total": loss_total,
            "loss_int": loss_int,
            "loss_grad": loss_grad,
            "loss_thermal": loss_thermal,
            "loss_sat": loss_sat,
        }
        return loss_total, loss_dict


if __name__ == "__main__":
    ir = torch.rand(2, 1, 256, 256)
    vis = torch.rand(2, 3, 256, 256)
    fused_logits = torch.randn(2, 1, 256, 256, requires_grad=True)
    outputs = {
        "fused": torch.sigmoid(fused_logits),
        "thermal_prior": torch.rand(2, 1, 256, 256),
        "sat_uncertainty": torch.rand(2, 1, 256, 256),
        "smoke_prior": torch.rand(2, 1, 256, 256),
        "w_ir": torch.rand(2, 1, 256, 256),
        "w_vis": torch.rand(2, 1, 256, 256),
    }

    criterion = DAMECFusionLoss(gradient_mode="directional")
    loss_total, loss_dict = criterion(outputs, ir, vis)

    for name, value in loss_dict.items():
        print(f"{name}: {value.item():.6f}")

    assert loss_total.dim() == 0
    assert not torch.isnan(loss_total)
    loss_total.backward()

    ramp = torch.linspace(0.0, 1.0, 32).view(1, 1, 1, 32).expand(1, 1, 16, 32)
    edge_outputs = {
        "fused": 1.0 - ramp,
        "thermal_prior": torch.zeros_like(ramp),
        "sat_uncertainty": torch.zeros_like(ramp),
        "smoke_prior": torch.zeros_like(ramp),
        "w_ir": torch.full_like(ramp, 0.5),
        "w_vis": torch.full_like(ramp, 0.5),
    }
    edge_vis = ramp.repeat(1, 3, 1, 1)
    _, magnitude_losses = DAMECFusionLoss(gradient_mode="magnitude")(edge_outputs, ramp, edge_vis)
    _, directional_losses = DAMECFusionLoss(gradient_mode="directional")(edge_outputs, ramp, edge_vis)
    assert directional_losses["loss_grad"] > magnitude_losses["loss_grad"] + 0.1

    weighted_w_ir = torch.full_like(ramp, 0.25, requires_grad=True)
    weighted_w_vis = torch.full_like(ramp, 0.75, requires_grad=True)
    weighted_fused = torch.zeros_like(ramp, requires_grad=True)
    weighted_outputs = {
        "fused": weighted_fused,
        "thermal_prior": torch.zeros_like(ramp),
        "sat_uncertainty": torch.zeros_like(ramp),
        "smoke_prior": torch.zeros_like(ramp),
        "w_ir": weighted_w_ir,
        "w_vis": weighted_w_vis,
    }
    weighted_vis = torch.ones_like(ramp).repeat(1, 3, 1, 1)
    weighted_loss, weighted_losses = DAMECFusionLoss(
        lambda_grad=0.0,
        lambda_thermal=0.0,
        lambda_sat=0.0,
        intensity_target="mec_weighted",
    )(weighted_outputs, torch.zeros_like(ramp), weighted_vis)
    assert torch.allclose(weighted_losses["loss_int"], torch.tensor(0.75))
    weighted_loss.backward()
    assert weighted_fused.grad is not None
    assert weighted_w_ir.grad is None
    assert weighted_w_vis.grad is None
    print("DAMECFusionLoss self-test passed.")
