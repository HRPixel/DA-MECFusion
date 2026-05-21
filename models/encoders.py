"""CNN encoders for DA-MECFusion V1."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Same-resolution residual block used by the V1 CNN encoder."""

    def __init__(self, channels: int, use_bn: bool = False) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")

        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(channels))
        layers.extend(
            [
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            ]
        )
        if use_bn:
            layers.append(nn.BatchNorm2d(channels))

        self.main = nn.Sequential(*layers)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.main(x))


class SimpleEncoder(nn.Module):
    """Simple same-resolution CNN encoder for one image modality."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        num_blocks: int = 2,
        base_channels: int = 64,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        stem: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
        ]
        if use_bn:
            stem.append(nn.BatchNorm2d(base_channels))
        stem.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        blocks = [
            ResidualBlock(base_channels, use_bn=use_bn)
            for _ in range(num_blocks)
        ]

        head: list[nn.Module] = [
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        ]
        if use_bn:
            head.append(nn.BatchNorm2d(out_channels))
        head.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*stem, *blocks, *head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}")
        return self.net(x)


if __name__ == "__main__":
    x = torch.rand(2, 1, 256, 256)
    model = SimpleEncoder(in_channels=1, out_channels=64)
    y = model(x)
    print(f"output: shape={tuple(y.shape)}, min={y.min().item():.6f}, max={y.max().item():.6f}")

    assert y.shape == (2, 64, 256, 256), y.shape
    loss = y.mean()
    loss.backward()
    print("SimpleEncoder self-test passed.")
