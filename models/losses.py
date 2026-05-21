"""Loss functions for DA-MECFusion V1."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.priors import rgb_to_y, sobel_gradient  # noqa: E402


class DAMECFusionLoss(nn.Module):
    """Basic V1 fusion loss without task-specific or smoke losses."""

    def __init__(
        self,
        lambda_int: float = 1.0,
        lambda_grad: float = 10.0,
        lambda_thermal: float = 1.0,
        lambda_sat: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_int = lambda_int
        self.lambda_grad = lambda_grad
        self.lambda_thermal = lambda_thermal
        self.lambda_sat = lambda_sat
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
        self._require_key(outputs, "w_ir")
        self._require_key(outputs, "w_vis")

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

        target_int = torch.maximum(ir, vis_y)
        loss_int = F.l1_loss(fused, target_int)

        grad_f = sobel_gradient(fused, eps=self.eps)
        grad_ir = sobel_gradient(ir, eps=self.eps)
        grad_vis = sobel_gradient(vis_y, eps=self.eps)
        target_grad = torch.maximum(grad_ir, grad_vis)
        loss_grad = F.l1_loss(grad_f, target_grad)

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

    criterion = DAMECFusionLoss()
    loss_total, loss_dict = criterion(outputs, ir, vis)

    for name, value in loss_dict.items():
        print(f"{name}: {value.item():.6f}")

    assert loss_total.dim() == 0
    assert not torch.isnan(loss_total)
    loss_total.backward()
    print("DAMECFusionLoss self-test passed.")
