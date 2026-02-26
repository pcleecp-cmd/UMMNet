import torch
import torch.nn as nn

from .ConvNeXtV2_Block import ConvNeXtV2_Block, LayerNorm
from .LRF_MSC import LRF_MSC
from .UDEC import UDEC
from .UMMF import UMMF


def conv1x1(in_c: int, out_c: int) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=1, bias=True)


class Encoder(nn.Module):
    def __init__(self, channels: int, n_blocks: int = 1, drop_path: float = 0.1):
        super().__init__()
        self.encoder_blocks = nn.Sequential(
            *[ConvNeXtV2_Block(channels, drop_path=drop_path) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_blocks(x)


class Down(nn.Sequential):
    def __init__(self, in_c: int, out_c: int):
        super().__init__(
            nn.BatchNorm2d(in_c),
            nn.Conv2d(in_c, out_c, kernel_size=2, stride=2),
        )


class UBlock(nn.Sequential):
    def __init__(
        self,
        in_c: int,
        mid_c: int,
        out_c: int,
        lrf_msc_dilations=(1, 12, 24),
    ):
        first_block = LRF_MSC(in_c, mid_c, dilations=lrf_msc_dilations)
        super().__init__(
            first_block,
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class UMMNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dims=[48, 96, 192, 384],
        blocks=[1, 1, 2, 1],
        drop_paths=[0.1, 0.1, 0.2, 0.2],
        ummf_dilations=(1, 2, 4),
        ummf_logvar_min: float = -8.0,
        ummf_logvar_max: float = 8.0,
        lrf_msc_dilations=(1, 12, 24),
        mc_dropout_p: float = 0.2,
        mc_dropout_samples_train: int = 1,
        mc_dropout_samples_eval: int = 10,
        udec_num_directions: int = 4,
        udec_delta: int = 1,
    ):
        super().__init__()
        self.dims = dims
        self.latest_analysis = {}

        # Stem
        self.stem1 = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=3, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=3, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )

        # Dual encoders
        self.enc1_1 = Encoder(dims[0], blocks[0], drop_paths[0])
        self.down1_1 = Down(dims[0], dims[1])
        self.enc1_2 = Encoder(dims[1], blocks[1], drop_paths[1])
        self.down1_2 = Down(dims[1], dims[2])
        self.enc1_3 = Encoder(dims[2], blocks[2], drop_paths[2])
        self.down1_3 = Down(dims[2], dims[3])
        self.enc1_4 = Encoder(dims[3], blocks[3], drop_paths[3])

        self.enc2_1 = Encoder(dims[0], blocks[0], drop_paths[0])
        self.down2_1 = Down(dims[0], dims[1])
        self.enc2_2 = Encoder(dims[1], blocks[1], drop_paths[1])
        self.down2_2 = Down(dims[1], dims[2])
        self.enc2_3 = Encoder(dims[2], blocks[2], drop_paths[2])
        self.down2_3 = Down(dims[2], dims[3])
        self.enc2_4 = Encoder(dims[3], blocks[3], drop_paths[3])

        def build_fusion(channels: int) -> UMMF:
            return UMMF(
                channels,
                first=True,
                fusion_variant="ummf",
                multi_scale_dilations=ummf_dilations,
                logvar_min=ummf_logvar_min,
                logvar_max=ummf_logvar_max,
                mc_dropout_p=mc_dropout_p,
                mc_dropout_samples_train=mc_dropout_samples_train,
                mc_dropout_samples_eval=mc_dropout_samples_eval,
            )

        self.bottleneck_fusion = build_fusion(dims[3])

        self.bottleneck_adjust_conv = nn.Sequential(
            nn.Conv2d(dims[3], dims[2], kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dims[2]),
            nn.ReLU(inplace=True),
        )

        self.ug_fusion_skip1 = build_fusion(dims[0])
        self.ug_fusion_skip2 = build_fusion(dims[1])
        self.ug_fusion_skip3 = build_fusion(dims[2])

        self.sam1 = UDEC(dims[2], num_directions=udec_num_directions, delta=udec_delta)
        self.sam2 = UDEC(dims[1], num_directions=udec_num_directions, delta=udec_delta)
        self.sam3 = UDEC(dims[0], num_directions=udec_num_directions, delta=udec_delta)

        self.dec3 = UBlock(
            dims[2] * 2,
            dims[2],
            dims[1],
            lrf_msc_dilations=lrf_msc_dilations,
        )
        self.dec2 = UBlock(
            dims[1] * 2,
            dims[1],
            dims[0],
            lrf_msc_dilations=lrf_msc_dilations,
        )
        self.dec1 = UBlock(
            dims[0] * 2,
            dims[0],
            dims[0],
            lrf_msc_dilations=lrf_msc_dilations,
        )

        self.out_conv = conv1x1(dims[0], num_classes)

    def _run_fusion(self, module, x0, m1, m2, save_analysis: bool):
        if isinstance(module, UMMF):
            return module(x0, m1, m2, save_uncertainty=save_analysis)
        return module(x0, m1, m2)

    def _run_sam(self, module, skip, dec, save_analysis: bool):
        if isinstance(module, UDEC):
            return module(skip, dec, save_intermediate=save_analysis)
        return module(skip, dec)

    def _cache_analysis(self):
        self.latest_analysis = {
            "ummf": {
                "bottleneck": (
                    self.bottleneck_fusion.get_visualizations()
                    if isinstance(self.bottleneck_fusion, UMMF)
                    else None
                ),
                "skip3": (
                    self.ug_fusion_skip3.get_visualizations()
                    if isinstance(self.ug_fusion_skip3, UMMF)
                    else None
                ),
                "skip2": (
                    self.ug_fusion_skip2.get_visualizations()
                    if isinstance(self.ug_fusion_skip2, UMMF)
                    else None
                ),
                "skip1": (
                    self.ug_fusion_skip1.get_visualizations()
                    if isinstance(self.ug_fusion_skip1, UMMF)
                    else None
                ),
            },
            "udec": {
                "stage3": self.sam1.get_direction_analysis(),
                "stage2": self.sam2.get_direction_analysis(),
                "stage1": self.sam3.get_direction_analysis(),
            },
        }

    def get_analysis_cache(self):
        return self.latest_analysis

    def forward(self, x: torch.Tensor, current_epoch=None, save_analysis: bool = False) -> torch.Tensor:
        x1_input = x[:, 0:1, :, :]
        x2_input = x[:, 1:2, :, :]

        s1 = self.stem1(x1_input)
        s2 = self.stem2(x2_input)

        f1_1 = self.enc1_1(s1)
        d1_1 = self.down1_1(f1_1)
        f1_2 = self.enc1_2(d1_1)
        d1_2 = self.down1_2(f1_2)
        f1_3 = self.enc1_3(d1_2)
        d1_3 = self.down1_3(f1_3)
        f1_4 = self.enc1_4(d1_3)

        f2_1 = self.enc2_1(s2)
        d2_1 = self.down2_1(f2_1)
        f2_2 = self.enc2_2(d2_1)
        d2_2 = self.down2_2(f2_2)
        f2_3 = self.enc2_3(d2_2)
        d2_3 = self.down2_3(f2_3)
        f2_4 = self.enc2_4(d2_3)

        bot_fused_ug = self._run_fusion(self.bottleneck_fusion, None, f1_4, f2_4, save_analysis)
        bot_fused_final = self.bottleneck_adjust_conv(bot_fused_ug)

        skip3_fused = self._run_fusion(self.ug_fusion_skip3, None, f1_3, f2_3, save_analysis)
        up3_sam_out = self._run_sam(self.sam1, skip3_fused, bot_fused_final, save_analysis)
        dec_out3 = self.dec3(up3_sam_out)

        skip2_fused = self._run_fusion(self.ug_fusion_skip2, None, f1_2, f2_2, save_analysis)
        up2_sam_out = self._run_sam(self.sam2, skip2_fused, dec_out3, save_analysis)
        dec_out2 = self.dec2(up2_sam_out)

        skip1_fused = self._run_fusion(self.ug_fusion_skip1, None, f1_1, f2_1, save_analysis)
        up1_sam_out = self._run_sam(self.sam3, skip1_fused, dec_out2, save_analysis)
        dec_out1 = self.dec1(up1_sam_out)
        if save_analysis:
            self._cache_analysis()

        return self.out_conv(dec_out1)


if __name__ == "__main__":
    model = UMMNet(num_classes=4, dims=[32, 64, 128, 256], blocks=[1, 1, 1, 1], drop_paths=[0.05, 0.05, 0.1, 0.1])
    dummy_input = torch.randn(2, 2, 224, 224)
    output = model(dummy_input)
    print("Model UMMNet instantiated and forward pass successful.")
    print("Output shape:", output.shape)
