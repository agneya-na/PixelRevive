#!/usr/bin/env python3
"""
run.py -- required entry point for the KLA benchmarking machine.

Usage:
    python run.py <input_dir> <output_dir>

Reads every .npy file in <input_dir>, runs RestoreNet (joint denoise + 2x
super-resolution) on it, and writes one restored .npy file per input file
to <output_dir> (created automatically if it does not exist), using the
exact same filename as the input.

Output arrays: float32, grayscale, shape (H, W), values clipped to [0, 1],
guaranteed free of NaN/Inf.

No internet access, API keys, or manual configuration required -- model
weights are loaded from ./models/restorenet_final.pt (bundled with this
submission) by default, and the device is auto-selected (CUDA if available,
else CPU).
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
    # tensors, which newer PyTorch versions (2.6+) would otherwise reject
    # under the new default weights_only=True.
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {"base_ch": 64, "n_res": 4, "scale": 2})
    model = RestoreNet(base_ch=cfg["base_ch"], n_res=cfg["n_res"], scale=cfg.get("scale", 2))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def _load_npy_gray01(path: str) -> np.ndarray:
    """Load a .npy file as a float32 2D grayscale array in [~0, ~1].

    Accepts (H, W), (H, W, 1), or (1, H, W) -- the singleton channel dim is
    squeezed. Integer arrays are assumed to be in [0, 255] and are scaled
    down; float arrays are trusted as already-normalized (NOT clipped here,
    since the degraded input can legitimately exceed 1.0 due to speckle --
    same convention as the rest of this codebase). NaN/Inf are sanitized.
    """
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


@torch.no_grad()
def restore_npy(model: RestoreNet, in_path: str, device: torch.device) -> np.ndarray:
    arr = _load_npy_gray01(in_path)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    # RestoreNet.forward() pads/crops to a multiple of 4 internally, so no
    # manual padding is needed here.
    pred, _ = model(x, stage=2)

    out = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="PixelRevive restoration -- KLA required entry point.")
    ap.add_argument("input_dir", help="folder of degraded .npy test files")
    ap.add_argument("output_dir", help="folder to write restored .npy files to")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to trained checkpoint (.pt)")
    ap.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'")
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

    print(f"device={device}  weights={args.weights}  files={len(files)}")

    times = []
    for f in files:
        t0 = time.time()
        restored = restore_npy(model, str(f), device)
        times.append(time.time() - t0)
        out_path = os.path.join(args.output_dir, f.name)  # same filename, incl. .npy
        np.save(out_path, restored)

    avg_ms = 1000 * sum(times) / len(times)
    print(f"done. {len(files)} files -> {args.output_dir}")
    print(f"avg inference time: {avg_ms:.1f} ms/file  (total {sum(times):.1f}s)")


if __name__ == "__main__":
    main()
