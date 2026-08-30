"""
RestoreNet v3: joint denoise + 2x super-resolution network.

v3 CHANGES vs v2 (all aimed at measurable PSNR/SSIM/LPIPS gains):
  - Configurable bottleneck depth (--n_middle, default 8). NAFNet's ablations
    show stacking extra blocks at the LOWEST resolution is the most
    cost-efficient capacity you can add -- compute there is 1/16th of full-res.
  - New `pre_sr` NAFBlock on the decoder features right before both heads.
    NAFBlocks are exact identities at init (beta=gamma=0), so this block is
    trainable capacity that starts as a no-op -- and, as a side effect, v2
    checkpoints still load and run (see run.py's flexible loader).

Backwards compatibility (unchanged public interface):
  - RestoreNet(base_ch=48, n_res=3, scale=2) still valid kwargs.
  - Checkpoints store config {"base_ch", "n_res", "n_middle", "scale"};
    v2 checkpoints lack "n_middle" -> defaults len(bottleneck)=n_res,
    which exactly matches the v2 layout, so old weights still work.
  - forward(x, stage=1) -> (restored, denoised_lr, restored_raw)
  - forward(x, stage=2) -> (restored, restored_raw)          [default]

v2 design (kept): NAFBlock core (SimpleGate + Simplified Channel Attention +
LayerNorm2d, per NAFNet, Chen et al., ECCV 2022), PixelUnshuffle/PixelShuffle
down/up-sampling (no ConvTranspose2d checkerboard artifacts, per Odena et al.,
Distill 2016), concat+1x1 skip fusion, auxiliary LR denoise head for Stage-1
pretraining, PixelShuffle x2 SR head, and a global bicubic residual so the
network only predicts a correction. Input is padded to a multiple of 4
internally and the output is cropped back automatically.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an NCHW tensor (NAFNet uses this
    instead of BatchNorm -- BN's batch statistics are a bad fit for
    restoration nets trained with small crops / variable-noise inputs)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """Parameter-free gating: split channels in half, multiply the halves.
    Replaces ReLU/GELU -- NAFNet's ablations show this alone beats standard
    activations for restoration quality at zero parameter cost."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet block: depthwise-conv branch with SimpleGate + Simplified
    Channel Attention (SCA), plus a SimpleGate FFN branch. Both branches use
    a learnable per-channel scale (beta/gamma, init 0) on the residual so the
    block starts as an exact identity and eases into the transform -- key for
    stability, and the reason `pre_sr` can be inserted without disturbing
    already-trained v2 checkpoints."""

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_ch, 1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)  # depthwise
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_ch // 2, c, 1)

        ffn_ch = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_ch, 1)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


def naf_stack(channels: int, n: int) -> nn.Sequential:
    return nn.Sequential(*[NAFBlock(channels) for _ in range(n)])


class DownsampleP2(nn.Module):
    """2x downsample via PixelUnshuffle (lossless space-to-depth) + 1x1
    projection. No stride-induced aliasing the way strided conv can have."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.proj = nn.Conv2d(in_ch * 4, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.unshuffle(x))


class UpsampleP2(nn.Module):
    """2x upsample via sub-pixel conv (1x1 conv + PixelShuffle). Avoids the
    checkerboard artifacts ConvTranspose2d is prone to."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch * 4, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.proj(x))


# --------------------------------------------------------------------------
# Main network
# --------------------------------------------------------------------------

class RestoreNet(nn.Module):
    def __init__(self, base_ch: int = 64, n_res: int = 4, n_middle: int = None, scale: int = 2):
        """
        base_ch:  channel width at input resolution (doubles per downsample).
        n_res:    NAFBlocks per encoder/decoder stage.
        n_middle: NAFBlocks in the bottleneck. NAFNet puts most of its depth
                  here because compute is 1/16th of full-res. If None
                  (e.g. when reconstructing from a v2 checkpoint config), it
                  falls back to n_res, which exactly matches v2's layout.
        scale:    must be 2 (dataset is a fixed 2x degradation).
        """
        super().__init__()
        assert scale == 2, "dataset only has a fixed 2x degradation (512->256, 256->128)"
        self.scale = scale
        if n_middle is None:
            n_middle = n_res  # v2-checkpoint backwards compatibility

        self.head = nn.Conv2d(1, base_ch, 3, padding=1)

        # Encoder
        self.enc1 = naf_stack(base_ch, n_res)
        self.down1 = DownsampleP2(base_ch, base_ch * 2)          # H -> H/2
        self.enc2 = naf_stack(base_ch * 2, n_res)
        self.down2 = DownsampleP2(base_ch * 2, base_ch * 4)      # H/2 -> H/4

        # Bottleneck (deeper in v3)
        self.bottleneck = naf_stack(base_ch * 4, n_middle)

        # Decoder (skip connections fused via concat+1x1)
        self.up2 = UpsampleP2(base_ch * 4, base_ch * 2)          # H/4 -> H/2
        self.fuse2 = nn.Conv2d(base_ch * 4, base_ch * 2, 1)
        self.dec2 = naf_stack(base_ch * 2, n_res)

        self.up1 = UpsampleP2(base_ch * 2, base_ch)              # H/2 -> H
        self.fuse1 = nn.Conv2d(base_ch * 2, base_ch, 1)
        self.dec1 = naf_stack(base_ch, n_res)

        # v3: one extra refinement block shared by both heads.
        # Identity at init (beta=gamma=0), so it adds capacity without
        # destabilizing early training or breaking v2 checkpoints.
        self.pre_sr = NAFBlock(base_ch)

        # Auxiliary denoise head: LR-resolution clean prediction (Stage-1 target,
        # and a light auxiliary loss during Stage 2 via --w_aux).
        self.denoise_head = nn.Conv2d(base_ch, 1, 3, padding=1)

        # SR head: H -> 2H via PixelShuffle, refined by one NAFBlock at full res.
        # (Module order kept identical to v2 so old state_dict keys still match.)
        self.sr_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            NAFBlock(base_ch),
            nn.Conv2d(base_ch, 1, 3, padding=1),
        )

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 4):
        _, _, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    def forward(self, x: torch.Tensor, stage: int = 2):
        """
        x: degraded LR input, shape (B, 1, H, W). Values may exceed [0, 1].
           H, W need NOT be multiples of 4 -- padded internally, cropped back.
        stage: 1 -> (restored, denoised_lr, restored_raw)
               2 -> (restored, restored_raw)
        """
        h0, w0 = x.shape[-2:]
        x, _, _ = self._pad_to_multiple(x, 4)

        f0 = self.head(x)
        e1 = self.enc1(f0)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        b = self.bottleneck(d2)

        u2 = self.up2(b)
        u2 = self.fuse2(torch.cat([u2, e2], dim=1))
        dec2 = self.dec2(u2)

        u1 = self.up1(dec2)
        u1 = self.fuse1(torch.cat([u1, e1], dim=1))
        dec1 = self.dec1(u1)

        feat = self.pre_sr(dec1)

        # Auxiliary LR denoise output (global residual on the input itself)
        denoised_lr = torch.clamp(x + self.denoise_head(feat), 0.0, 1.0)
        denoised_lr = denoised_lr[:, :, :h0, :w0]

        # Main SR output (global residual on a bicubic-upsampled input)
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        residual = self.sr_head(feat)
        restored_raw = base + residual          # pre-clamp, used by the range-penalty loss
        restored = torch.clamp(restored_raw, 0.0, 1.0)

        restored_raw = restored_raw[:, :, : h0 * 2, : w0 * 2]
        restored = restored[:, :, : h0 * 2, : w0 * 2]

        if stage == 1:
            return restored, denoised_lr, restored_raw
        return restored, restored_raw


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Shape sanity check (no_grad: just verifying output shapes across sizes,
    # including a non-multiple-of-4 one to exercise the internal pad/crop path).
    m = RestoreNet()
    with torch.no_grad():
        for h, w in [(128, 128), (256, 256), (101, 130)]:
            x = torch.rand(1, 1, h, w) * 1.15  # simulate speckle pushing values > 1
            out, denoised_lr, raw = m(x, stage=1)
            assert out.shape == (1, 1, h * 2, w * 2), out.shape
            assert denoised_lr.shape == x.shape, denoised_lr.shape
            assert out.min() >= 0.0 and out.max() <= 1.0
            out2, raw2 = m(x, stage=2)
            assert out2.shape == (1, 1, h * 2, w * 2), out2.shape
            print(f"input {tuple(x.shape)} -> output {tuple(out.shape)}  OK")

    # Trainability check: gradients must flow end-to-end through
    # LayerNorm2d / SimpleGate / the pixel-shuffle heads.
    x = torch.rand(1, 1, 32, 32)
    out, raw = m(x, stage=2)
    out.sum().backward()
    assert m.head.weight.grad is not None and torch.isfinite(m.head.weight.grad).all()
    assert m.pre_sr.beta.grad is not None
    assert m.sr_head[-1].weight.grad is not None
    print("backward() OK, gradients finite")

    print("params (base_ch=48, n_res=3, v2-small):", count_params(RestoreNet(base_ch=48, n_res=3)))
    print("params (base_ch=64, n_res=4, n_middle=4, = v2 size):",
          count_params(RestoreNet(base_ch=64, n_res=4, n_middle=4)))
    print("params (base_ch=64, n_res=4, n_middle=8, v3 default):",
          count_params(RestoreNet(base_ch=64, n_res=4, n_middle=8)))
    print("ALL MODEL SELF-TESTS PASSED")
