import torch
import torch.nn as nn
import torch.nn.functional as F


class UDEC(nn.Module):
    SUPPORTED_DIRECTION_COUNTS = {4, 8}

    def __init__(
        self,
        C: int,
        num_directions: int = 4,
        delta: int = 1,
        context_dilations=(1, 2, 4),
    ):
        super().__init__()
        if num_directions not in self.SUPPORTED_DIRECTION_COUNTS:
            raise ValueError(
                f"num_directions must be one of {sorted(self.SUPPORTED_DIRECTION_COUNTS)}, got {num_directions}"
            )
        if delta < 1:
            raise ValueError(f"delta must be >= 1, got {delta}")
        if len(context_dilations) != 3:
            raise ValueError(
                f"context_dilations must have 3 integers, got {context_dilations}"
            )

        self.num_directions = int(num_directions)
        self.delta = int(delta)
        d0, d1, d2 = [int(d) for d in context_dilations]

        self.directions = nn.ModuleList(
            [
                nn.Conv2d(2 * C, 2 * C, 3, 1, d0, groups=2 * C, dilation=d0),
                nn.Conv2d(2 * C, 2 * C, 3, 1, d1, groups=2 * C, dilation=d1),
                nn.Conv2d(2 * C, 2 * C, 3, 1, d2, groups=2 * C, dilation=d2),
            ]
        )

        self.direction_att = nn.Sequential(
            nn.Conv2d(2 * C, C, 1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, self.num_directions, 1),
            nn.Softmax(dim=1),
        )

        self.enhance = nn.Sequential(
            nn.Conv2d(2 * C, C, 3, 1, 1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, 1, 1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.before_correction = None
        self.after_correction = None
        self.direction_weights = None
        self.direction_entropy = None

    def _direction_specs(self):
        d = self.delta
        labels_and_shifts = [
            ("up", (-d, 0)),
            ("down", (d, 0)),
            ("left", (0, -d)),
            ("right", (0, d)),
        ]
        if self.num_directions == 8:
            labels_and_shifts.extend(
                [
                    ("up_left", (-d, -d)),
                    ("up_right", (-d, d)),
                    ("down_left", (d, -d)),
                    ("down_right", (d, d)),
                ]
            )
        return labels_and_shifts

    def forward(self, high_res, low_res, save_intermediate: bool = False):
        _, _, H, W = high_res.shape
        low = F.interpolate(low_res, size=(H, W), mode="bilinear", align_corners=False)

        if save_intermediate:
            self.before_correction = high_res.detach()

        dir_features = [conv(torch.cat([high_res, low], 1)) for conv in self.directions]
        dir_features = sum(dir_features)

        gamma = self.direction_att(dir_features)

        aligned = 0.0
        for idx, (_, shift) in enumerate(self._direction_specs()):
            aligned = aligned + gamma[:, idx : idx + 1] * torch.roll(
                low, shifts=shift, dims=(2, 3)
            )

        enhanced = self.enhance(torch.cat([high_res, aligned], 1))
        fused = high_res + self.alpha * (aligned + enhanced)

        if save_intermediate:
            self.after_correction = fused.detach()
            self.direction_weights = gamma.detach()
            self.direction_entropy = (
                -(gamma * torch.log(gamma + 1e-6)).sum(dim=1, keepdim=True).detach()
            )

        return torch.cat([high_res, fused], dim=1)

    def get_correction_effect(self):
        return {
            "before_correction": self.before_correction,
            "after_correction": self.after_correction,
        }

    def get_direction_analysis(self):
        return {
            "gamma": self.direction_weights,
            "entropy": self.direction_entropy,
            "direction_labels": [name for name, _ in self._direction_specs()],
            "num_directions": self.num_directions,
            "delta": self.delta,
        }
