import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

try:
    from model import RestoreNet
except ImportError:
    try:
        from models.model import RestoreNet
    except ImportError:
        sys.exit("ERROR: Could not import RestoreNet from model.py or models/model.py")


def _pad_to_multiple(x: torch.Tensor, multiple: int = 4):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w


def load_model(weights_path: str, device: torch.device) -> RestoreNet:
    if not os.path.isfile(weights_path):
        sys.exit(f"ERROR: Model weights not found at '{weights_path}'")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {"base_ch": 48, "n_res": 3, "scale": 2})
    model = RestoreNet(base_ch=cfg["base_ch"], n_res=cfg["n_res"], scale=cfg.get("scale", 2))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


@torch.no_grad()
def restore_npy(model: RestoreNet, file_path: str, device: torch.device) -> np.ndarray:
    raw = np.load(file_path)
    if raw.ndim == 3:
        raw = raw.squeeze()
    if raw.ndim != 2:
        raise ValueError(f"{file_path}: expected a 2D grayscale array after squeeze, got shape {raw.shape}")

    raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)

    # dtype-based rescale -- NOT value-based. Degraded inputs can legitimately
    # exceed 1.0 (speckle noise pushes values out of range); only rescale if
    # the array is actually stored as integer pixel data (0-255).
    if np.issubdtype(raw.dtype, np.integer):
        arr = raw.astype(np.float32) / 255.0
    else:
        arr = raw.astype(np.float32)

    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    x_pad, pad_h, pad_w = _pad_to_multiple(x, 4)

    pred, _ = model(x_pad, stage=2)

    h, w = arr.shape
    pred = pred[:, :, : h * 2, : w * 2]
    out = pred.squeeze(0).squeeze(0).cpu().numpy()

    # Safety checks: NaN/Inf filter & range [0.0, 1.0]
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser(description="KLA Restoration Solution Entry Point")
    parser.add_argument("input_dir", help="Path to input directory containing .npy files")
    parser.add_argument("output_dir", help="Path to output directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "models", "restorenet_final.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(script_dir, "restorenet_final.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(weights_path, device)

    input_files = sorted(Path(args.input_dir).glob("*.npy"))
    if not input_files:
        sys.exit(f"ERROR: No .npy files found in {args.input_dir}")

    for f in input_files:
        restored_arr = restore_npy(model, str(f), device)
        out_file = Path(args.output_dir) / f.name
        np.save(out_file, restored_arr)

    print(f"Successfully processed {len(input_files)} .npy files into {args.output_dir}")


if __name__ == "__main__":
    main()
