"""Fusion decoder for DA-MECFusion V1."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Same-resolution residual block for the fusion decoder."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        return self.act2(out + residual)


class FusionDecoder(nn.Module):
    """Decode fused feature maps into a normalized single-channel image."""

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        num_blocks: int = 2,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if hidden_channels <= 1:
            raise ValueError("hidden_channels must be greater than 1")
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")

        mid_channels = hidden_channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            *[ResidualBlock(hidden_channels) for _ in range(num_blocks)],
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, fused_feat: torch.Tensor) -> torch.Tensor:
        if fused_feat.dim() != 4:
            raise ValueError(
                f"Expected fused_feat with shape [B,C,H,W], got {tuple(fused_feat.shape)}"
            )
        return self.net(fused_feat)


if __name__ == "__main__":
    fused_feat = torch.randn(2, 64, 256, 256, requires_grad=True)
    model = FusionDecoder(in_channels=64)
    fused = model(fused_feat)

    print(
        f"fused: shape={tuple(fused.shape)}, "
        f"min={fused.min().item():.6f}, max={fused.max().item():.6f}, "
        f"mean={fused.mean().item():.6f}"
    )

    assert fused.shape == (2, 1, 256, 256), fused.shape
    assert fused.min().item() >= 0.0
    assert fused.max().item() <= 1.0

    loss = fused.mean()
    loss.backward()
    print("FusionDecoder self-test passed.")
