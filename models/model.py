"""
RestoreNet: joint denoise + 2x super-resolution network.

Design (matches PixelRevive_KLA_PS01.pptx, slides 3-4):
  - Encoder: residual conv blocks + strided downsampling.
  - Bottleneck: residual blocks.
  - Decoder: transposed-conv upsampling + skip connections from encoder.
  - Auxiliary "denoise head": predicts a clean image at INPUT resolution
    from the decoder features. Used only in Stage 1 pretraining (see train.py).
  - SR head: PixelShuffle x2 upsampling block that takes the network to
    2x the input resolution (128->256 or 256->512).
  - Global residual learning: the network predicts a correction on top of
    a bicubic-upsampled version of the input, rather than the raw pixels.
    This is standard in SR literature (VDSR/EDSR-style) and is what makes
    "residual learning separates noise from content" (slide 3) concrete
    rather than just a claim.
  - Output is clamped to [0, 1] to match the ground-truth range, since the
    degraded input can exceed [0, 1] due to speckle noise (dataset note).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return x + out


def res_stack(channels: int, n: int) -> nn.Sequential:
    return nn.Sequential(*[ResBlock(channels) for _ in range(n)])


class RestoreNet(nn.Module):
    def __init__(self, base_ch: int = 48, n_res: int = 3, scale: int = 2):
        super().__init__()
        assert scale == 2, "dataset only has a fixed 2x degradation (512->256, 256->128)"
        self.scale = scale

        self.head = nn.Conv2d(1, base_ch, 3, padding=1)

        # Encoder
        self.enc1 = res_stack(base_ch, n_res)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)   # H -> H/2

        self.enc2 = res_stack(base_ch * 2, n_res)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)  # H/2 -> H/4

        # Bottleneck
        self.bottleneck = res_stack(base_ch * 4, n_res)

        # Decoder (mirrors encoder, skip connections)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)  # H/4 -> H/2
        self.dec2 = res_stack(base_ch * 2, n_res)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)  # H/2 -> H
        self.dec1 = res_stack(base_ch, n_res)

        # Auxiliary denoise head: LR-resolution clean prediction (Stage 1 target)
        self.denoise_head = nn.Conv2d(base_ch, 1, 3, padding=1)

        # SR head: H -> 2H via PixelShuffle
        self.sr_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, stage: int = 2):
        """
        x: degraded LR input, shape (B, 1, H, W). Values may exceed [0, 1].
        stage: 1 -> also return the LR-resolution denoise prediction (for
               Stage-1 pretraining loss). 2 -> normal full forward pass.

        Returns:
          restored: (B, 1, 2H, 2W) tensor in [0, 1]
          denoised_lr: (B, 1, H, W) tensor in [0, 1], only meaningful/returned
                       when stage == 1 (auxiliary output, cheap to compute always)
        """
        f0 = self.head(x)
        e1 = self.enc1(f0)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        b = self.bottleneck(d2)

        u2 = self.up2(b) + e2
        dec2 = self.dec2(u2)

        u1 = self.up1(dec2) + e1
        dec1 = self.dec1(u1)

        # Auxiliary LR denoise output (global residual on the input itself)
        denoised_lr = torch.clamp(x + self.denoise_head(dec1), 0.0, 1.0)

        # Main SR output (global residual on a bicubic-upsampled input)
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        residual = self.sr_head(dec1)
        restored_raw = base + residual          # pre-clamp, used by the range-penalty loss
        restored = torch.clamp(restored_raw, 0.0, 1.0)

        if stage == 1:
            return restored, denoised_lr, restored_raw
        return restored, restored_raw


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # quick shape sanity check
    for h in (128, 256):
        m = RestoreNet()
        x = torch.rand(2, 1, h, h) * 1.15  # simulate speckle pushing values > 1
        out, denoised_lr, raw = m(x, stage=1)
        assert out.shape == (2, 1, h * 2, h * 2), out.shape
        assert denoised_lr.shape == x.shape, denoised_lr.shape
        assert out.min() >= 0.0 and out.max() <= 1.0
        print(f"input {tuple(x.shape)} -> output {tuple(out.shape)}  OK")
    print("params:", count_params(RestoreNet()))
