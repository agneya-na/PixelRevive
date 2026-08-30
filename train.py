"""
Training script v3 (SELF-CONTAINED -- dataset loading/pairing and all loss
functions are inlined below; the only external project dependency is
models/model.py, which run.py also imports so it must stay standalone).

v3 changes vs v2 (each one targets a reported metric):
  - PSNR is now computed in validation alongside L1/SSIM, and training keeps
    FOUR best checkpoints: best-by-SSIM and best-by-PSNR (raw + EMA each).
  - EMA is RESET before Stage 2 by default -- stage-1's EMA average lags far
    behind the fine-tuned weights and was diluting stage-2 improvements.
    (--keep_ema_stage2 restores the old behavior.)
  - New FrequencyLoss: L1 distance between rFFT magnitudes of pred and GT.
    Frequency-space losses are a standard trick for recovering the fine,
    repetitive high-frequency structure of semiconductor imagery that pixel
    losses underweight (--w_fft, default 0.05; set 0 to disable).
  - New auxiliary LR-denoise loss DURING Stage 2 (--w_aux, default 0.1) so
    the denoiser half of the network stays anchored while the SR head trains.
  - Optional Stage 3 (--epochs_stage3, default 0): a short pixel-loss-only
    polish at low LR. Standard final step for squeezing PSNR after the
    perceptual-heavy stage.
  - New model capacity knob --n_middle (bottleneck depth, default 8).
  - TF32 matmul + cudnn.benchmark enabled for free speed on Ampere+ GPUs.

Kept from v2: 80/10/10 split; Stage-1 denoise pretraining against an
anti-alias-downsampled "clean LR" target; AdamW + linear warmup -> per-step
cosine decay per stage; EMA (decay 0.999) with restorenet_final.pt storing
EMA weights; AMP on CUDA; gradient clipping; loader accepting .npy AND
image files; speckle/gaussian re-degrade augmentation.

USAGE (Colab/Kaggle GPU):
    python train.py --gt_dir /path/to/train/GT --lr_dir /path/to/train/NoisyLR \
        --out_dir checkpoints --epochs_stage1 15 --epochs_stage2 80 \
        --epochs_stage3 5 --batch_size 16 --crop_lr 128 --n_middle 8
"""
import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ssim as ssim_fn
import torchvision.models as tvm

from models.model import RestoreNet, count_params


# ==========================================================================
# Dataset (formerly dataset.py -- inlined so this file is self-contained)
# ==========================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NPY_EXTS = {".npy"}
ALL_EXTS = IMG_EXTS | NPY_EXTS


def _list_images(folder: str) -> List[str]:
    return sorted(
        f.name for f in Path(folder).iterdir()
        if f.suffix.lower() in ALL_EXTS
    )


def _strip_tag(name: str) -> str:
    """Strip common GT/LR/HR/noisy/degraded tags and extension to get a matchable stem."""
    stem = Path(name).stem
    stem = re.sub(r"(?i)[_\-]?(gt|hr|lr|noisy|degraded|clean|ground[_\-]?truth)$", "", stem)
    stem = re.sub(r"(?i)^(gt|hr|lr|noisy|degraded|clean)[_\-]?", "", stem)
    return stem.lower()


def pair_files(gt_dir: str, lr_dir: str) -> List[Tuple[str, str]]:
    """
    Pairing strategy (first one that covers all files wins):
      1. Exact same filename in both folders.
      2. Same stem after stripping common tags ("0001_HR" <-> "0001_LR").
      3. Fallback: same sorted position (prints a loud warning + first 3 pairs).
    """
    gt_files = _list_images(gt_dir)
    lr_files = _list_images(lr_dir)
    if not gt_files or not lr_files:
        raise RuntimeError(f"No images found. gt_dir has {len(gt_files)}, lr_dir has {len(lr_files)}.")

    common = sorted(set(gt_files) & set(lr_files))
    if len(common) == len(gt_files) == len(lr_files):
        return [(g, g) for g in common]

    gt_by_stem = {_strip_tag(f): f for f in gt_files}
    lr_by_stem = {_strip_tag(f): f for f in lr_files}
    stems = sorted(set(gt_by_stem) & set(lr_by_stem))
    if len(stems) == len(gt_files) == len(lr_files):
        return [(gt_by_stem[s], lr_by_stem[s]) for s in stems]

    n = min(len(gt_files), len(lr_files))
    print(
        f"[dataset] WARNING: could not match filenames 1:1 between\n"
        f"  gt_dir={gt_dir} ({len(gt_files)} files)\n"
        f"  lr_dir={lr_dir} ({len(lr_files)} files)\n"
        f"  Falling back to sorted-order pairing for the first {n} files.\n"
        f"  First 3 pairs: {list(zip(gt_files[:3], lr_files[:3]))}\n"
        f"  >>> VERIFY these are actually the same scenes before trusting training. <<<"
    )
    return list(zip(gt_files[:n], lr_files[:n]))


def _load_gray01(path: str) -> np.ndarray:
    """Load as float32 2D grayscale array. Supports .npy (trusted as
    already-normalized float -- NOT clipped, degraded arrays may exceed 1.0
    due to speckle; integer arrays assumed [0,255] and scaled) and standard
    images (PIL grayscale, /255). NaN/Inf sanitized. (H,W,1)/(1,H,W) squeezed."""
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        if arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
            elif arr.shape[0] == 1:
                arr = arr[0]
            else:
                raise ValueError(f"{path}: expected a grayscale array, got shape {arr.shape}")
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected a 2D array after squeeze, got shape {arr.shape}")
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        return arr
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def synthetic_redegrade(lr: np.ndarray, p: float = 0.3) -> np.ndarray:
    """Extra speckle + gaussian noise on an already-degraded LR patch."""
    if random.random() > p:
        return lr
    out = lr.copy()
    if random.random() < 0.5:
        sigma = random.uniform(0.02, 0.08)
        out = out + out * np.random.normal(0, sigma, out.shape).astype(np.float32)  # speckle
    if random.random() < 0.5:
        sigma = random.uniform(0.01, 0.05)
        out = out + np.random.normal(0, sigma, out.shape).astype(np.float32)  # gaussian
    return out.astype(np.float32)


class PairedRestorationDataset(Dataset):
    def __init__(self, gt_dir, lr_dir, pairs=None, crop_lr=128, train=True, redegrade_p=0.3):
        self.gt_dir = gt_dir
        self.lr_dir = lr_dir
        self.pairs = pairs if pairs is not None else pair_files(gt_dir, lr_dir)
        self.crop_lr = crop_lr
        self.train = train
        self.redegrade_p = redegrade_p

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_name, lr_name = self.pairs[idx]
        gt = _load_gray01(os.path.join(self.gt_dir, gt_name))
        lr = _load_gray01(os.path.join(self.lr_dir, lr_name))

        if gt.shape[0] != 2 * lr.shape[0] or gt.shape[1] != 2 * lr.shape[1]:
            raise ValueError(
                f"Resolution mismatch for pair ({gt_name}, {lr_name}): "
                f"GT={gt.shape}, LR={lr.shape}. Expected GT to be exactly 2x LR in both dims."
            )

        if self.train:
            lr, gt = self._augment(lr, gt)

        lr_t = torch.from_numpy(lr).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt).unsqueeze(0).float()
        return lr_t, gt_t

    def _augment(self, lr: np.ndarray, gt: np.ndarray):
        h, w = lr.shape
        c = min(self.crop_lr, h, w)
        # RestoreNet downsamples 4x internally (PixelUnshuffle twice), so the
        # LR crop's H and W must divide by 4.
        c = max(4, (c // 4) * 4)
        top = random.randint(0, h - c) if h > c else 0
        left = random.randint(0, w - c) if w > c else 0
        lr_c = lr[top:top + c, left:left + c]
        gt_c = gt[top * 2:(top + c) * 2, left * 2:(left + c) * 2]

        if random.random() < 0.5:
            lr_c, gt_c = np.fliplr(lr_c).copy(), np.fliplr(gt_c).copy()
        if random.random() < 0.5:
            lr_c, gt_c = np.flipud(lr_c).copy(), np.flipud(gt_c).copy()
        k = random.randint(0, 3)
        if k:
            lr_c, gt_c = np.rot90(lr_c, k).copy(), np.rot90(gt_c, k).copy()

        lr_c = synthetic_redegrade(lr_c, p=self.redegrade_p)
        return lr_c, gt_c


def make_splits(gt_dir, lr_dir, val_frac=0.1, test_frac=0.1, seed=42):
    """80/10/10 split. Returns 3 lists of (gt_name, lr_name)."""
    pairs = pair_files(gt_dir, lr_dir)
    rng = random.Random(seed)
    pairs = pairs[:]
    rng.shuffle(pairs)
    n = len(pairs)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    test = pairs[:n_test]
    val = pairs[n_test:n_test + n_val]
    train = pairs[n_test + n_val:]
    return train, val, test


# ==========================================================================
# Losses (formerly losses.py -- inlined so this file is self-contained)
# ==========================================================================

class CharbonnierLoss(nn.Module):
    """Smooth L1 variant: sqrt((pred-target)^2 + eps^2), eps=1e-3."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class SobelEdgeLoss(nn.Module):
    """L1 distance between Sobel-gradient magnitudes of pred and target --
    directly rewards sharp, well-placed edges (what LPIPS punishes most)."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3).contiguous()
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _grad_mag(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred, target):
        return F.l1_loss(self._grad_mag(pred), self._grad_mag(target))


class FrequencyLoss(nn.Module):
    """L1 on rFFT magnitudes (ortho-normalized). Pixel losses are dominated by
    low frequencies; this term re-weights the high-frequency energy where the
    repetitive wafer structure lives -> better SSIM on fine detail.
    Casts to float32 internally so it's AMP-safe (FFT isn't autocast-friendly)."""

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        fp = torch.fft.rfft2(pred, norm="ortho").abs()
        ft = torch.fft.rfft2(target, norm="ortho").abs()
        return F.l1_loss(fp, ft)


class VGGPerceptual(nn.Module):
    """VGG16 relu2_2 + relu3_3 feature distance. Grayscale input is repeated
    to 3 channels; VGG weights frozen."""

    def __init__(self, layers=("relu2_2", "relu3_3"), layer_weights=(0.5, 1.0)):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        cuts = {"relu2_2": 9, "relu3_3": 16}
        self.layers = list(layers)
        self.layer_weights = list(layer_weights)
        self.cuts = {l: cuts[l] for l in self.layers}
        max_cut = max(self.cuts.values())
        self.slice = nn.Sequential(*[vgg[i] for i in range(max_cut)]).eval()
        for p in self.slice.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _features(self, x3):
        feats = {}
        h = x3
        wanted = set(self.cuts.values())
        for i, layer in enumerate(self.slice):
            h = layer(h)
            if (i + 1) in wanted:
                for name, cut in self.cuts.items():
                    if cut == i + 1:
                        feats[name] = h
        return feats

    def forward(self, x, y):
        x3 = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        y3 = (y.repeat(1, 3, 1, 1) - self.mean) / self.std
        with torch.no_grad():
            fy = self._features(y3)
        fx = self._features(x3)
        loss = x.new_zeros(())
        for name, w in zip(self.layers, self.layer_weights):
            loss = loss + w * F.l1_loss(fx[name], fy[name])
        return loss


class CombinedLoss(nn.Module):
    """total = w_l1*Charbonnier + w_ssim*(1-SSIM) + w_perc*VGG-perceptual
               + w_range*RangePenalty + w_edge*SobelEdge + w_fft*FrequencyLoss"""

    def __init__(self, w_l1=1.0, w_ssim=0.5, w_perc=0.15, w_range=0.05,
                 w_edge=0.05, w_fft=0.05, use_perceptual=True, use_charbonnier=True):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_perc = w_perc
        self.w_range = w_range
        self.w_edge = w_edge
        self.w_fft = w_fft
        self.use_perceptual = use_perceptual
        self.pixel_loss = CharbonnierLoss() if use_charbonnier else nn.L1Loss()
        self.edge_loss = SobelEdgeLoss()
        self.fft_loss = FrequencyLoss()
        if use_perceptual:
            self.vgg = VGGPerceptual()

    def range_penalty(self, raw):
        over = F.relu(raw - 1.0)
        under = F.relu(-raw)
        return (over.pow(2).mean() + under.pow(2).mean())

    def forward(self, pred, pred_raw, target):
        pixel = self.pixel_loss(pred, target)
        ssim_val = ssim_fn(pred, target, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val
        range_loss = self.range_penalty(pred_raw)
        edge_loss = self.edge_loss(pred, target)

        total = (self.w_l1 * pixel + self.w_ssim * ssim_loss
                 + self.w_range * range_loss + self.w_edge * edge_loss)
        logs = {"pixel": pixel.item(), "ssim_loss": ssim_loss.item(),
                "range": range_loss.item(), "edge": edge_loss.item()}

        if self.w_fft > 0:
            fft = self.fft_loss(pred, target)
            total = total + self.w_fft * fft
            logs["fft"] = fft.item()

        if self.use_perceptual:
            perc = self.vgg(pred, target)
            total = total + self.w_perc * perc
            logs["perc"] = perc.item()

        logs["total"] = total.item()
        return total, logs


# ==========================================================================
# Training
# ==========================================================================

def get_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


class EMA:
    """Exponential moving average of model parameters + buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow


def build_scheduler(opt, warmup_steps: int, total_steps: int):
    """Linear warmup -> cosine decay to 0, stepped once per batch."""
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        prog = min(1.0, prog)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _psnr_per_image(pred, target):
    """Per-image PSNR in dB, both inputs already in [0,1]."""
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


@torch.no_grad()
def validate(model, loader, device):
    """Returns (L1, SSIM, PSNR) on the validation split."""
    model.eval()
    tot_l1, tot_ssim, tot_psnr, n = 0.0, 0.0, 0.0, 0
    for lr, gt in loader:
        lr, gt = lr.to(device), gt.to(device)
        pred, _ = model(lr, stage=2)
        tot_l1 += F.l1_loss(pred, gt).item()
        tot_ssim += ssim_fn(pred, gt, data_range=1.0, size_average=True).item()
        tot_psnr += _psnr_per_image(pred, gt).mean().item()
        n += 1
    model.train()
    n = max(n, 1)
    return tot_l1 / n, tot_ssim / n, tot_psnr / n


def run_stage1(model, train_loader, val_loader, device, epochs, lr, out_dir,
               warmup_steps, grad_clip, use_amp, ema: EMA):
    print(f"\n=== Stage 1: denoise pretraining ({epochs} epochs) ===")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * max(1, len(train_loader))
    sched = build_scheduler(opt, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0
        for lr_img, gt in train_loader:
            lr_img, gt = lr_img.to(device), gt.to(device)
            gt_lr_target = F.interpolate(gt, scale_factor=0.5, mode="area")

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, denoised_lr, _ = model(lr_img, stage=1)
                loss = F.l1_loss(denoised_lr, gt_lr_target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            running += loss.item()
        val_l1, val_ssim, val_psnr = validate(model, val_loader, device)
        dt = time.time() - t0
        cur_lr = sched.get_last_lr()[0]
        print(f"[stage1] epoch {ep}/{epochs}  train_L1={running/len(train_loader):.4f}  "
              f"val_L1={val_l1:.4f}  val_SSIM={val_ssim:.4f}  val_PSNR={val_psnr:.2f}  "
              f"lr={cur_lr:.2e}  ({dt:.1f}s)")

        if val_l1 < best_val:
            best_val = val_l1
            save_checkpoint(model, out_dir, "stage1_best.pt")

    save_checkpoint(model, out_dir, "stage1_final.pt")


def run_stage2(model, train_loader, val_loader, device, epochs, lr, out_dir, loss_weights,
               use_perceptual, warmup_steps, grad_clip, use_amp, ema: EMA,
               w_aux=0.1, pixel_only=False, tag="stage2", save_best=True):
    """Joint fine-tune. pixel_only=True -> Stage-3 polish with Charbonnier only."""
    if pixel_only:
        pixel_crit = CharbonnierLoss()
        print(f"\n=== {tag}: pixel-only polish ({epochs} epochs) ===")
    else:
        loss_fn = CombinedLoss(use_perceptual=use_perceptual, **loss_weights).to(device)
        print(f"\n=== {tag}: joint fine-tuning ({epochs} epochs) ===")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * max(1, len(train_loader))
    sched = build_scheduler(opt, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_ssim, best_psnr = -float("inf"), -float("inf")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        running = {}
        for lr_img, gt in train_loader:
            lr_img, gt = lr_img.to(device), gt.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                # stage=1 gives us all three outputs; denoised_lr is reused by
                # the auxiliary loss so the denoiser stays anchored.
                restored, denoised_lr, restored_raw = model(lr_img, stage=1)
                if pixel_only:
                    loss = pixel_crit(restored, gt)
                    logs = {"pixel": loss.item(), "total": loss.item()}
                else:
                    loss, logs = loss_fn(restored, restored_raw, gt)
                    if w_aux > 0:
                        gt_lr_target = F.interpolate(gt, scale_factor=0.5, mode="area")
                        aux = F.l1_loss(denoised_lr, gt_lr_target)
                        loss = loss + w_aux * aux
                        logs["aux"] = aux.item()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v

        n = len(train_loader)
        log_str = "  ".join(f"{k}={v/n:.4f}" for k, v in running.items())
        val_l1, val_ssim, val_psnr = validate(model, val_loader, device)
        dt = time.time() - t0
        cur_lr = sched.get_last_lr()[0]
        print(f"[{tag}] epoch {ep}/{epochs}  {log_str}  val_L1={val_l1:.4f}  "
              f"val_SSIM={val_ssim:.4f}  val_PSNR={val_psnr:.2f}  lr={cur_lr:.2e}  ({dt:.1f}s)")

        if save_best:
            # Track the two headline metrics separately -- the best-SSIM and
            # best-PSNR epochs are usually NOT the same epoch.
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                save_checkpoint(model, out_dir, "restorenet_best.pt")
                save_checkpoint(model, out_dir, "restorenet_best_ema.pt", state_dict=ema.state_dict())
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(model, out_dir, "restorenet_best_psnr.pt")
                save_checkpoint(model, out_dir, "restorenet_best_psnr_ema.pt", state_dict=ema.state_dict())


def save_checkpoint(model, out_dir, name, state_dict=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save({
        "model_state": state_dict if state_dict is not None else model.state_dict(),
        "config": model.config,
    }, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True, help="folder with clean/GT images (.npy or images)")
    ap.add_argument("--lr_dir", required=True, help="folder with degraded/LR images (.npy or images)")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--epochs_stage1", type=int, default=15)
    ap.add_argument("--epochs_stage2", type=int, default=80)
    ap.add_argument("--epochs_stage3", type=int, default=0,
                    help="optional pixel-only polish epochs at the end (try 5) -- pure PSNR squeeze")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--crop_lr", type=int, default=128)
    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--n_res", type=int, default=4, help="NAFBlocks per encoder/decoder stage")
    ap.add_argument("--n_middle", type=int, default=8,
                    help="NAFBlocks in the bottleneck -- cheapest capacity (compute is 1/16 res)")
    ap.add_argument("--lr1", type=float, default=2e-4, help="stage 1 learning rate")
    ap.add_argument("--lr2", type=float, default=1e-4, help="stage 2 learning rate")
    ap.add_argument("--lr3", type=float, default=3e-5, help="stage 3 (polish) learning rate")
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--keep_ema_stage2", action="store_true",
                    help="do NOT reset EMA between stage 1 and 2 (v2 behavior; worse)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_perceptual", action="store_true")
    ap.add_argument("--w_l1", type=float, default=1.0)
    ap.add_argument("--w_ssim", type=float, default=0.5)
    ap.add_argument("--w_perc", type=float, default=0.15)
    ap.add_argument("--w_range", type=float, default=0.05)
    ap.add_argument("--w_edge", type=float, default=0.05)
    ap.add_argument("--w_fft", type=float, default=0.05, help="frequency-domain loss weight; 0 disables")
    ap.add_argument("--w_aux", type=float, default=0.1,
                    help="stage-2 auxiliary LR-denoise loss weight; 0 disables")
    ap.add_argument("--no_charbonnier", action="store_true")
    args = ap.parse_args()

    # Reproducibility + free speed (TF32 matmuls / cudnn autotune on Ampere+).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = get_device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp
    print("device:", device, " amp:", use_amp)

    train_pairs, val_pairs, test_pairs = make_splits(args.gt_dir, args.lr_dir, seed=args.seed)
    print(f"split -> train {len(train_pairs)}  val {len(val_pairs)}  test {len(test_pairs)}")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "test_split.json"), "w") as f:
        json.dump(test_pairs, f, indent=2)
    print(f"test split saved to {args.out_dir}/test_split.json")

    train_ds = PairedRestorationDataset(args.gt_dir, args.lr_dir, pairs=train_pairs,
                                         crop_lr=args.crop_lr, train=True)
    val_ds = PairedRestorationDataset(args.gt_dir, args.lr_dir, pairs=val_pairs, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True,
                               pin_memory=(device.type == "cuda"),
                               persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = RestoreNet(base_ch=args.base_ch, n_res=args.n_res, n_middle=args.n_middle).to(device)
    model.config = {"base_ch": args.base_ch, "n_res": args.n_res,
                    "n_middle": args.n_middle, "scale": model.scale}
    print("model params:", count_params(model))
    ema = EMA(model, decay=args.ema_decay)

    t_start = time.time()
    loss_weights = dict(w_l1=args.w_l1, w_ssim=args.w_ssim, w_perc=args.w_perc,
                        w_range=args.w_range, w_edge=args.w_edge, w_fft=args.w_fft,
                        use_charbonnier=not args.no_charbonnier)

    run_stage1(model, train_loader, val_loader, device, args.epochs_stage1, args.lr1, args.out_dir,
               args.warmup_steps, args.grad_clip, use_amp, ema)

    # v3: reset EMA before joint fine-tuning -- the stage-1 average lags too
    # far behind and dilutes stage-2 improvements in the final EMA weights.
    if not args.keep_ema_stage2:
        print("resetting EMA for stage 2 (use --keep_ema_stage2 to disable)")
        ema = EMA(model, decay=args.ema_decay)

    run_stage2(model, train_loader, val_loader, device, args.epochs_stage2, args.lr2, args.out_dir,
               loss_weights, use_perceptual=not args.no_perceptual,
               warmup_steps=args.warmup_steps, grad_clip=args.grad_clip, use_amp=use_amp,
               ema=ema, w_aux=args.w_aux, tag="stage2")

    if args.epochs_stage3 > 0:
        run_stage2(model, train_loader, val_loader, device, args.epochs_stage3, args.lr3,
                   args.out_dir, loss_weights, use_perceptual=False,
                   warmup_steps=args.warmup_steps, grad_clip=args.grad_clip, use_amp=use_amp,
                   ema=ema, pixel_only=True, tag="stage3", save_best=False)

    save_checkpoint(model, args.out_dir, "restorenet_final_raw.pt")
    save_checkpoint(model, args.out_dir, "restorenet_final.pt", state_dict=ema.state_dict())
    print("restorenet_final.pt holds the EMA weights (what run.py will load).")

    total_time = time.time() - t_start
    final_path = os.path.join(args.out_dir, "restorenet_final.pt")
    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    print(f"\n=== DONE ===")
    print(f"total training time: {total_time/60:.1f} min")
    print(f"final checkpoint: {final_path}  ({size_mb:.1f} MB)")
    print("artifacts: restorenet_final.pt (EMA), restorenet_final_raw.pt,")
    print("           restorenet_best[_ema].pt (best val SSIM),")
    print("           restorenet_best_psnr[_ema].pt (best val PSNR), test_split.json")


if __name__ == "__main__":
    main()
