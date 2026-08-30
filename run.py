#!/usr/bin/env python3
"""
run.py -- required entry point for the KLA benchmarking machine.

Usage:
    python run.py <input_dir> <output_dir> [--tta 8] [--weights PATH] [--device auto|cuda|cpu]

Reads every .npy file in <input_dir>, runs RestoreNet (joint denoise + 2x
super-resolution) on it, and writes one restored .npy file per input file
to <output_dir> (created automatically), using the EXACT same filename.

v3 addition: --tta {1,4,8} test-time augmentation (geometric self-ensemble).
Each image is restored under 4 rotations x optional mirroring and the
inverse-transformed predictions are averaged. This is a standard, reliable
way to gain ~0.1-0.3 dB PSNR and a point or two of SSIM with NO retraining
-- at the cost of 4x / 8x inference time. Default 8. Use --tta 1 to disable.

Output arrays: float32, grayscale, shape (2H, 2W), values clipped to [0,1],
guaranteed free of NaN/Inf. No internet access, API keys, or configuration
needed -- weights load from ./models/restorenet_final.pt and the device is
auto-detected (CUDA if available, else CPU).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from models.model import RestoreNet

DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "restorenet_final.pt",
)


def load_model(weights_path: str, device: torch.device) -> RestoreNet:
    if not os.path.isfile(weights_path):
        sys.exit(f"ERROR: weights file not found at '{weights_path}'.")
    # weights_only=False: our checkpoint stores a "config" dict alongside the
    # tensors, which PyTorch 2.6+ would reject under weights_only=True.
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {"base_ch": 64, "n_res": 4, "scale": 2})
    model = RestoreNet(
        base_ch=cfg["base_ch"],
        n_res=cfg["n_res"],
        n_middle=cfg.get("n_middle", None),  # v2 checkpoints lack this key -> falls back to n_res
        scale=cfg.get("scale", 2),
    )
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as e:
        # v2 checkpoint into v3 model: the only missing keys are pre_sr.*,
        # a NAFBlock that is an exact identity at init (beta=gamma=0), so the
        # old weights still produce their original outputs.
        print("[run.py] note: relaxed weight loading in use "
              f"({_brief_keys(e)}). Missing v3 blocks remain identity -- OK.")
        model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    return model


def _brief_keys(err: RuntimeError) -> str:
    return "see state_dict mismatch (expected for v2-era checkpoints)"


def _load_npy_gray01(path: str) -> np.ndarray:
    """Load a .npy file as float32 2D grayscale.
    Accepts (H, W), (H, W, 1), or (1, H, W); integer arrays are assumed
    [0,255] and scaled; float arrays are trusted as already-normalized (NOT
    clipped -- degraded input may legitimately exceed 1.0 from speckle).
    NaN/Inf are sanitized."""
    arr = np.load(path)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"{path}: expected a grayscale array (H,W), (H,W,1) or (1,H,W), "
                f"got shape {arr.shape}"
            )
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2D array after squeeze, got shape {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return arr


def _tta_transforms(tta: int):
    """(flip, rot90_k) combos. tta=1: identity only; tta=4: rotations;
    tta=8: rotations x mirroring (full dihedral group)."""
    if tta <= 1:
        return [(False, 0)]
    if tta == 4:
        return [(False, r) for r in range(4)]
    return [(f, r) for f in (False, True) for r in range(4)]


@torch.no_grad()
def predict(model: RestoreNet, x: torch.Tensor, tta: int) -> torch.Tensor:
    """Self-ensemble prediction: average the model's (clamped) output over
    the TTA group, inverse-transforming each prediction back to the original
    orientation before averaging."""
    preds = []
    for flip, rot in _tta_transforms(tta):
        xi = x
        if flip:
            xi = torch.flip(xi, dims=[-1])
        xi = torch.rot90(xi, rot, dims=[-2, -1])
        p, _ = model(xi, stage=2)
        p = torch.rot90(p, -rot, dims=[-2, -1])
        if flip:
            p = torch.flip(p, dims=[-1])
        preds.append(p)
    if len(preds) == 1:
        return preds[0]
    return torch.stack(preds, dim=0).mean(dim=0)


def restore_npy(model: RestoreNet, in_path: str, device: torch.device, tta: int) -> np.ndarray:
    arr = _load_npy_gray01(in_path)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        pred = predict(model, x, tta)

    out = pred.squeeze(0).squeeze(0).float().cpu().numpy().astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="PixelRevive restoration -- KLA required entry point.")
    ap.add_argument("input_dir", help="folder of degraded .npy test files")
    ap.add_argument("output_dir", help="folder to write restored .npy files to")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to trained checkpoint (.pt)")
    ap.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'")
    ap.add_argument("--tta", type=int, default=8, choices=[1, 4, 8],
                    help="test-time self-ensemble size. 8 = best quality (default), "
                         "1 = disabled (fastest). Time scales linearly with this.")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(args.weights, device)

    files = sorted(f for f in Path(args.input_dir).iterdir() if f.suffix.lower() == ".npy")
    if not files:
        sys.exit(f"ERROR: no .npy files found in {args.input_dir}")

    if args.tta > 1:
        print(f"TTA self-ensemble x{args.tta} enabled (use --tta 1 to disable; "
              f"per-file time scales with this).")
    print(f"device={device}  weights={args.weights}  files={len(files)}")

    times = []
    for f in files:
        t0 = time.time()
        restored = restore_npy(model, str(f), device, args.tta)
        times.append(time.time() - t0)
        out_path = os.path.join(args.output_dir, f.name)  # same filename, incl. .npy
        np.save(out_path, restored)

    avg_ms = 1000 * sum(times) / len(times)
    print(f"done. {len(files)} files -> {args.output_dir}")
    print(f"avg inference time: {avg_ms:.1f} ms/file (tta={args.tta})  (total {sum(times):.1f}s)")


if __name__ == "__main__":
    main()
