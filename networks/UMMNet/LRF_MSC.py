import torch
import torch.nn as nn


class LRF_MSC(nn.Module):
    """Large receptive-field multi-scale context with configurable dilations."""

    def __init__(self, C_in: int, C_out: int = None, dilations=(1, 12, 24)):
        super().__init__()
        C_out = C_in if C_out is None else C_out
        if len(dilations) != 3:
            raise ValueError(f"dilations must have 3 integers, got {dilations}")
        d0, d1, d2 = [int(d) for d in dilations]

        self.dw3 = nn.Conv2d(
            C_in, C_in, kernel_size=3, stride=1, padding=d0, dilation=d0, groups=C_in
        )
        self.dw3d = nn.Conv2d(
            C_in, C_in, kernel_size=3, stride=1, padding=d1, dilation=d1, groups=C_in
        )
        self.dw3d2 = nn.Conv2d(
            C_in, C_in, kernel_size=3, stride=1, padding=d2, dilation=d2, groups=C_in
        )
        self.proj = nn.Conv2d(3 * C_in, C_out, kernel_size=1)

    def forward(self, x):
        y1 = self.dw3(x)
        y2 = self.dw3d(x)
        y3 = self.dw3d2(x)
        return self.proj(torch.cat([y1, y2, y3], dim=1))
