import torch
import torch.nn as nn
import torch.nn.functional as F


class UMMF(nn.Module):
    SUPPORTED_VARIANTS = {
        "ummf",
        "concat",
        "cross_att_only",
        "uncertainty_only",
        "mc_dropout_uq",
    }

    def __init__(
        self,
        C: int,
        first: bool = False,
        fusion_variant: str = "ummf",
        multi_scale_dilations=(1, 2, 4),
        logvar_min: float = -8.0,
        logvar_max: float = 8.0,
        mc_dropout_p: float = 0.2,
        mc_dropout_samples_train: int = 1,
        mc_dropout_samples_eval: int = 10,
    ):
        super().__init__()
        self.first = first
        self.fusion_variant = fusion_variant
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.mc_dropout_p = float(mc_dropout_p)
        self.mc_dropout_samples_train = int(mc_dropout_samples_train)
        self.mc_dropout_samples_eval = int(mc_dropout_samples_eval)
        if self.logvar_min >= self.logvar_max:
            raise ValueError(
                f"logvar_min must be < logvar_max, got {self.logvar_min} >= {self.logvar_max}"
            )
        if len(multi_scale_dilations) != 3:
            raise ValueError(
                f"multi_scale_dilations must have 3 integers, got {multi_scale_dilations}"
            )
        d0, d1, d2 = [int(d) for d in multi_scale_dilations]

        if self.fusion_variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported fusion_variant={self.fusion_variant}. "
                f"Available: {sorted(self.SUPPORTED_VARIANTS)}"
            )

        # Multi-scale feature extraction.
        self.multi_scale = nn.ModuleList(
            [
                nn.Conv2d(C, C, 3, 1, d0, groups=C, dilation=d0),
                nn.Conv2d(C, C, 3, 1, d1, groups=C, dilation=d1),
                nn.Conv2d(C, C, 3, 1, d2, groups=C, dilation=d2),
            ]
        )

        # Cross-modal attention.
        self.cross_modal_att = nn.Sequential(
            nn.Conv2d(2 * C, C, 1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, 2, 1),
            nn.Softmax(dim=1),
        )

        # Feature enhancement.
        self.enhance = nn.Sequential(
            nn.Conv2d(2 * C, C, 3, 1, 1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, 1, 1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # Reliability prediction via log-variance.
        self.logvar = nn.Conv2d(2 * C, 2, 1)

        # Shared pre-fusion branch.
        self.pre = nn.Sequential(
            nn.Conv2d(2 * C, C, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(C if first else C * 2, C, 3, 1, 1)

        # Optional visualization caches.
        self.uncertainty_m1 = None
        self.uncertainty_m2 = None
        self.attention_weights = None

    def _compute_mc_dropout_uncertainty(self, feat: torch.Tensor, num_samples: int) -> torch.Tensor:
        if num_samples <= 1:
            return torch.zeros(
                feat.shape[0], 1, feat.shape[2], feat.shape[3], device=feat.device, dtype=feat.dtype
            )

        samples = []
        for _ in range(num_samples):
            dropped = F.dropout2d(feat, p=self.mc_dropout_p, training=True)
            samples.append(dropped)
        stacked = torch.stack(samples, dim=0)  # [T, B, C, H, W]
        return stacked.var(dim=0, unbiased=False).mean(dim=1, keepdim=True)  # [B, 1, H, W]

    def forward(self, x0, m1, m2, save_uncertainty: bool = False):
        # Baseline 1: concat fusion.
        if self.fusion_variant == "concat":
            return self.pre(torch.cat([m1, m2], 1))

        eps = 1e-6
        m1_scales = [conv(m1) for conv in self.multi_scale]
        m2_scales = [conv(m2) for conv in self.multi_scale]
        m1_multi = sum(m1_scales)
        m2_multi = sum(m2_scales)

        att_weights = self.cross_modal_att(torch.cat([m1_multi, m2_multi], 1))
        att_weights = torch.nan_to_num(att_weights, nan=0.5, posinf=1.0, neginf=0.0)
        att_weights = att_weights / att_weights.sum(dim=1, keepdim=True).clamp(min=eps)
        m1_att = m1_multi * att_weights[:, 0:1]
        m2_att = m2_multi * att_weights[:, 1:2]
        enhanced = self.enhance(torch.cat([m1_att, m2_att], 1))

        lv1 = lv2 = None
        w1 = w2 = None

        if self.fusion_variant in {"ummf", "uncertainty_only"}:
            lv1, lv2 = self.logvar(torch.cat([m1, m2], 1)).chunk(2, 1)
            lv1 = torch.clamp(lv1, min=self.logvar_min, max=self.logvar_max)
            lv2 = torch.clamp(lv2, min=self.logvar_min, max=self.logvar_max)
            w1, w2 = torch.exp(-lv1), torch.exp(-lv2)
        elif self.fusion_variant == "mc_dropout_uq":
            mc_samples = self.mc_dropout_samples_train if self.training else self.mc_dropout_samples_eval
            uq1 = self._compute_mc_dropout_uncertainty(m1, mc_samples)
            uq2 = self._compute_mc_dropout_uncertainty(m2, mc_samples)
            w1 = 1.0 / (uq1 + eps)
            w2 = 1.0 / (uq2 + eps)
            lv1 = torch.log(uq1 + eps)
            lv2 = torch.log(uq2 + eps)

        if w1 is not None and w2 is not None:
            w1 = torch.nan_to_num(w1, nan=1.0, posinf=1e4, neginf=0.0)
            w2 = torch.nan_to_num(w2, nan=1.0, posinf=1e4, neginf=0.0)

        if save_uncertainty:
            self.uncertainty_m1 = lv1.detach() if lv1 is not None else None
            self.uncertainty_m2 = lv2.detach() if lv2 is not None else None
            self.attention_weights = att_weights.detach()

        if self.fusion_variant == "cross_att_only":
            fused = m1_att + m2_att
            fuse = self.pre(torch.cat([m1, m2], 1)) + fused + enhanced
        elif self.fusion_variant in {"uncertainty_only", "mc_dropout_uq"}:
            denom = (w1 + w2).clamp(min=eps)
            fused = (w1 * m1 + w2 * m2) / denom
            fuse = self.pre(torch.cat([m1, m2], 1)) + fused
        else:
            # Full UMMF: cross-attention + logvar reliability gating.
            denom = (w1 + w2).clamp(min=eps)
            fused = (w1 * m1_att + w2 * m2_att) / denom
            fuse = self.pre(torch.cat([m1, m2], 1)) + fused + enhanced

        fuse = torch.nan_to_num(fuse, nan=0.0, posinf=1e4, neginf=-1e4)
        if not self.first:
            fuse = self.out(torch.cat([x0, fuse], 1))
            fuse = torch.nan_to_num(fuse, nan=0.0, posinf=1e4, neginf=-1e4)
        return fuse

    def get_visualizations(self):
        if self.uncertainty_m1 is None or self.uncertainty_m2 is None:
            uncertainty_map = None
        else:
            uncertainty_map = 0.5 * (self.uncertainty_m1 + self.uncertainty_m2)
        return {
            "uncertainty_m1": self.uncertainty_m1,
            "uncertainty_m2": self.uncertainty_m2,
            "uncertainty_map": uncertainty_map,
            "attention_weights": self.attention_weights,
        }
