"""
Training script for RestoreNet.

- 80/10/10 train/val/test split of the paired training set.
- Stage 1: pretrain the encoder-decoder as a plain denoiser (target is the
  real GT anti-alias-downsampled to LR size, since we don't have a clean-LR
  image directly). Stage 2: unfreeze everything and train the full
  denoise+SR pipeline end to end against the real GT, with the combined
  Charbonnier+SSIM+perceptual+edge+range loss (see losses.py).
- AdamW, linear warmup into per-step cosine decay, gradient clipping,
  mixed precision on CUDA, EMA of the weights (decay 0.999).
  restorenet_final.pt holds the EMA weights; restorenet_final_raw.pt keeps
  the raw last-iterate weights alongside it. See CHANGES.md for the reasoning
  behind each of these.

USAGE (run this on Colab/Kaggle with a GPU, not on a laptop CPU):
    python train.py --gt_dir /path/to/train/GT --lr_dir /path/to/train/NoisyLR \
        --out_dir checkpoints --epochs_stage1 15 --epochs_stage2 40 \
        --batch_size 16 --crop_lr 128

--gt_dir / --lr_dir must point at the two folders inside your unzipped
train.zip. Check the actual folder names first (see README) -- this script
does not guess them.
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import RestoreNet, count_params
from losses import CombinedLoss
from dataset import PairedRestorationDataset, make_splits


def get_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


class EMA:
    """Exponential moving average of model parameters + buffers.
    update() every step; state_dict() gives you a checkpoint-ready dict."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
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
        import math
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    tot_l1, tot_ssim, n = 0.0, 0.0, 0
    from pytorch_msssim import ssim as ssim_fn
    for lr, gt in loader:
        lr, gt = lr.to(device), gt.to(device)
        pred, _ = model(lr, stage=2)
        tot_l1 += F.l1_loss(pred, gt).item()
        tot_ssim += ssim_fn(pred, gt, data_range=1.0, size_average=True).item()
        n += 1
    model.train()
    return tot_l1 / max(n, 1), tot_ssim / max(n, 1)


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
        val_l1, val_ssim = validate(model, val_loader, device)
        dt = time.time() - t0
        cur_lr = sched.get_last_lr()[0]
        print(f"[stage1] epoch {ep}/{epochs}  train_L1={running/len(train_loader):.4f}  "
              f"val_L1={val_l1:.4f}  val_SSIM={val_ssim:.4f}  lr={cur_lr:.2e}  ({dt:.1f}s)")

        if val_l1 < best_val:
            best_val = val_l1
            save_checkpoint(model, out_dir, "stage1_best.pt")

    save_checkpoint(model, out_dir, "stage1_final.pt")


def run_stage2(model, train_loader, val_loader, device, epochs, lr, out_dir, loss_weights,
               use_perceptual, warmup_steps, grad_clip, use_amp, ema: EMA):
    print(f"\n=== Stage 2: joint fine-tuning ({epochs} epochs) ===")
    loss_fn = CombinedLoss(use_perceptual=use_perceptual, **loss_weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * max(1, len(train_loader))
    sched = build_scheduler(opt, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val = -float("inf")  # track by val SSIM, higher is better

    for ep in range(1, epochs + 1):
        t0 = time.time()
        running = {}
        for lr_img, gt in train_loader:
            lr_img, gt = lr_img.to(device), gt.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred, pred_raw = model(lr_img, stage=2)
                loss, logs = loss_fn(pred, pred_raw, gt)
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
        val_l1, val_ssim = validate(model, val_loader, device)
        dt = time.time() - t0
        cur_lr = sched.get_last_lr()[0]
        print(f"[stage2] epoch {ep}/{epochs}  {log_str}  val_L1={val_l1:.4f}  "
              f"val_SSIM={val_ssim:.4f}  lr={cur_lr:.2e}  ({dt:.1f}s)")

        if val_ssim > best_val:
            best_val = val_ssim
            save_checkpoint(model, out_dir, "restorenet_best.pt")
            save_checkpoint(model, out_dir, "restorenet_best_ema.pt", state_dict=ema.state_dict())

    save_checkpoint(model, out_dir, "restorenet_final_raw.pt")
    save_checkpoint(model, out_dir, "restorenet_final.pt", state_dict=ema.state_dict())
    print("restorenet_final.pt holds the EMA weights (what eval.py/compute_metrics.py will load).")
    print("restorenet_final_raw.pt holds the plain last-iterate weights, kept for comparison.")


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
    ap.add_argument("--gt_dir", required=True, help="folder with clean/GT images")
    ap.add_argument("--lr_dir", required=True, help="folder with degraded/LR images")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--epochs_stage1", type=int, default=15)
    ap.add_argument("--epochs_stage2", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--crop_lr", type=int, default=128)
    ap.add_argument("--base_ch", type=int, default=64,
                     help="v2 default raised 48->64: NAFBlock is cheaper per-block than the "
                          "old ResBlock (see model.py), so this still nets FEWER total params "
                          "than the original base_ch=48 model while adding capacity.")
    ap.add_argument("--n_res", type=int, default=4, help="NAFBlocks per stage (v2 default 3->4)")
    ap.add_argument("--lr1", type=float, default=2e-4, help="stage 1 learning rate")
    ap.add_argument("--lr2", type=float, default=1e-4, help="stage 2 learning rate")
    ap.add_argument("--warmup_steps", type=int, default=300, help="linear LR warmup steps, each stage")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="gradient-norm clip value")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision even on CUDA")
    ap.add_argument("--no_perceptual", action="store_true", help="disable VGG loss (faster, needs no internet)")
    ap.add_argument("--w_l1", type=float, default=1.0, help="weight on the Charbonnier pixel loss")
    ap.add_argument("--w_ssim", type=float, default=0.5)
    ap.add_argument("--w_perc", type=float, default=0.15, help="v2 default raised 0.1->0.15 (see README)")
    ap.add_argument("--w_range", type=float, default=0.05)
    ap.add_argument("--w_edge", type=float, default=0.05, help="Sobel edge-sharpness loss weight (new)")
    ap.add_argument("--no_charbonnier", action="store_true", help="use plain L1 instead of Charbonnier")
    args = ap.parse_args()

    device = get_device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp
    print("device:", device, " amp:", use_amp)

    train_pairs, val_pairs, test_pairs = make_splits(args.gt_dir, args.lr_dir)
    print(f"split -> train {len(train_pairs)}  val {len(val_pairs)}  test {len(test_pairs)}")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "test_split.json"), "w") as f:
        json.dump(test_pairs, f, indent=2)
    print(f"test split saved to {args.out_dir}/test_split.json -- reuse this for compute_metrics.py "
          f"so metrics are computed on data the model never trained on.")

    train_ds = PairedRestorationDataset(args.gt_dir, args.lr_dir, pairs=train_pairs,
                                         crop_lr=args.crop_lr, train=True)
    val_ds = PairedRestorationDataset(args.gt_dir, args.lr_dir, pairs=val_pairs, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True,
                               pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = RestoreNet(base_ch=args.base_ch, n_res=args.n_res).to(device)
    model.config = {"base_ch": args.base_ch, "n_res": args.n_res, "scale": model.scale}
    print("model params:", count_params(model))
    ema = EMA(model, decay=args.ema_decay)

    t_start = time.time()
    run_stage1(model, train_loader, val_loader, device, args.epochs_stage1, args.lr1, args.out_dir,
               args.warmup_steps, args.grad_clip, use_amp, ema)
    run_stage2(model, train_loader, val_loader, device, args.epochs_stage2, args.lr2, args.out_dir,
               loss_weights=dict(w_l1=args.w_l1, w_ssim=args.w_ssim, w_perc=args.w_perc,
                                  w_range=args.w_range, w_edge=args.w_edge,
                                  use_charbonnier=not args.no_charbonnier),
               use_perceptual=not args.no_perceptual,
               warmup_steps=args.warmup_steps, grad_clip=args.grad_clip, use_amp=use_amp, ema=ema)
    total_time = time.time() - t_start

    final_path = os.path.join(args.out_dir, "restorenet_final.pt")
    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    print(f"\n=== DONE ===")
    print(f"total training time: {total_time/60:.1f} min")
    print(f"final checkpoint: {final_path}  ({size_mb:.1f} MB)")
    print(f"-> copy these two numbers into slide 7 (Training Time, Model Size)")


if __name__ == "__main__":
    main()
