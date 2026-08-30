# PixelRevive - AI-Based Restoration of Degraded Images

Joint denoising + 2x super-resolution network (`RestoreNet`, a NAFNet-style encoder-decoder) for restoring 
degraded grayscale semiconductor images. KLA hackathon submission.

**This README alone is enough to clone this repo and run inference** Every command below is copy-pasteable as-is.

---

## Repository structure

```
<team_name>/
├── README.md                    # Component 1 -- this file
├── run.py                       # Component 2 -- REQUIRED evaluation script
├── train.py                     # Component 3 -- training script (self-contained:
│                                 #   dataset loading/pairing and all loss functions
│                                 #   are inlined here, no separate dataset.py/losses.py)
├── requirements.txt             # Component 6 -- pip dependencies
├── models/
│   ├── model.py                 # RestoreNet architecture -- imported as
│   │                             #   `from models.model import RestoreNet` by both
│   │                             #   run.py and train.py
│   └── restorenet_final.pt      # Component 4 -- trained weights
└── restored_test_outputs.zip       # Component 5 -- our model's output on the test set
    ├── 000000.png
    ├── 000001.png
    └── ...
```

---

## 0. Prerequisites

- Python 3.9–3.12 (any of these work).
- `pip` (comes with Python).
- A GPU is **optional** — `run.py` (inference) auto-detects CUDA and falls
  back to CPU. Training (`train.py`) also runs on CPU but is very slow;
  use Colab/Kaggle for that step if you don't have a GPU.

Check your Python version first:
```bash
python3 --version
```
If this prints `Python 3.9` or higher, you're good to go.

---

## 1. Setup

Clone the repo and install dependencies:
```bash
git clone <this_repo_url>
cd <team_name>
pip install -r requirements.txt
```
This is the **only** step that may need internet access (to download the
`torch`/`torchvision` packages). Once installed, running `run.py` itself
never needs internet, an API key, or any extra download - the trained
weights load from the local `models/` folder and the compute device
(CUDA or CPU) is picked automatically.

**If `pip install` fails or `pip` isn't recognized**, try:
```bash
python3 -m pip install -r requirements.txt
```

---

## 2. Component 2 — Evaluation script: `run.py` (runs as-is, no edits)

This is the script KLA will run directly on their benchmarking machine.

### How to run it
```bash
python run.py <input_dir> <output_dir>
```
Replace `<input_dir>` and `<output_dir>` with real folder paths. Concrete
example:
```bash
python run.py test_data/NoisyLR restored_test_outputs
```
Here `test_data/NoisyLR` is a folder full of `.npy` files you want restored,
and `restored_test_outputs` is where the restored `.npy` files will be
written. **You do not need to create `<output_dir>` yourself** — the script
creates it automatically if it doesn't already exist.

### What it does, step by step
1. Loads the trained weights from `models/restorenet_final.pt`.
2. Auto-picks the device: GPU (CUDA) if available, otherwise CPU.
3. Reads every `.npy` file inside `<input_dir>`.
4. Runs each one through the model (denoise + 2x super-resolution).
5. Saves each result to `<output_dir>`, using **the exact same filename**
   as the input file (e.g. `test_0001.npy` in → `test_0001.npy` out).
6. Prints how many files it processed and the average inference time per
   file — this is the number that goes on the "Inference Time" slide.

### Optional flags
| Flag | Default | What it does |
|---|---|---|
| `--weights <path>` | `models/restorenet_final.pt` | Use a different checkpoint file |
| `--device` | `auto` | Force `cuda` or `cpu` instead of auto-detecting |
| `--tta {1,4,8}` | `8` | Test-time self-ensemble (rotations x mirroring, averaged). 8 = best quality (default); each step multiplies per-file time. Use `--tta 1` to disable. |

Example using both:
```bash
python run.py test_data/NoisyLR restored_test_outputs --device cuda
```

### Input / output format (for reference — you don't need to do anything, `run.py` handles this automatically)
**Input:** grayscale `.npy`, shape `(H,W)`, `(H,W,1)`, or `(1,H,W)`. Integer
arrays are assumed `[0,255]`; float arrays are assumed already normalized to
roughly `[0,1]` (values may exceed 1.0 — that's expected, from speckle
noise, and is handled correctly).

**Output:** `float32` `.npy`, shape `(2H, 2W)` (exactly double the input's
height and width), values clipped to `[0,1]`, guaranteed free of NaN/Inf.

---

## 3. Component 3 — Training script: `train.py` (reproduces our model from scratch)

You only need this if you want to retrain the model yourself. If you just
want to run inference with our already-trained weights, skip to Section 2.

```bash
python train.py \
  --gt_dir /path/to/train/GT --lr_dir /path/to/train/NoisyLR \
  --out_dir checkpoints \
  --epochs_stage1 15 --epochs_stage2 80 --epochs_stage3 5 \
  --batch_size 16 --crop_lr 128 --base_ch 64 --n_res 4 --n_middle 8

```
- `--gt_dir` / `--lr_dir` must point at the two folders inside the unzipped
  training data (the clean/ground-truth images and the degraded/noisy-LR
  images respectively). **Check the actual folder names in your dataset
  first** — this script does not guess them, it needs the real path.
- Accepts either `.npy` files or standard images (`.png`/`.jpg`/etc.) in
  those folders.
- Two-stage training: Stage 1 pretrains a denoiser, Stage 2 fine-tunes the
  full denoise+SR pipeline end-to-end with a combined Charbonnier + SSIM +
  VGG-perceptual + Sobel-edge + range-penalty loss.
- Needs a GPU (Colab/Kaggle recommended) — CPU will run but is very slow
  for real dataset sizes.
- Writes `checkpoints/restorenet_final.pt` (EMA weights — this is the file
  `run.py` expects) plus a few intermediate/best checkpoints and
  `checkpoints/test_split.json`.
- See every available flag and its default: `python train.py --help`.

**After training finishes**, copy the new checkpoint into the `models/`
folder so `run.py` picks it up:
```bash
cp checkpoints/restorenet_final.pt models/restorenet_final.pt
```

---

## 4. Component 4 — Trained model weights: `models/restorenet_final.pt`

The final checkpoint (EMA weights), loadable directly by `run.py` and by
`train.py`'s checkpoint format: `{"model_state": ..., "config": {"base_ch",
"n_res", "scale"}}`. ~14MB, committed directly to the repo — no Git LFS
needed. It lives in `models/` right next to `model.py`.

You don't need to do anything with this file to run inference — `run.py`
finds it automatically at `models/restorenet_final.pt` (a path relative to
`run.py`'s own location, so it works regardless of which folder you run the
command from, as long as you keep the repo's folder structure intact).

---

## 5. Component 5 — Restored Test Outputs: `restored_test_outputs/`

**What this folder is:** the actual `.npy` files our model produced when we
ran it on the KLA test set. This is proof-of-output, not something a
reviewer needs to regenerate — but they can, using the exact command below,
to confirm our numbers.

### Naming rule (important)
Every file in `restored_test_outputs/` keeps **the exact same filename** as
the input file it came from. If the KLA test set has a file called
`Test_0042.npy`, our output for it is named `restored_test_outputs/Test_0042.npy`
— never renamed, re-numbered, or given a different extension. This is
enforced automatically by `run.py` (see step 5 in Section 2 above), not
something we did by hand.

### How this folder was generated
We ran the exact same evaluation command described in Section 2, pointed at
KLA's provided test set:
```bash
python run.py <path_to_KLA_test_set> restored_test_outputs
```
That's it — one command, no manual steps, no per-file editing. Whatever
`<path_to_KLA_test_set>` was on our machine, every `.npy` inside it got a
restored counterpart with the same name inside `restored_test_outputs/`.

### How a reviewer can reproduce/verify it
1. Get the official KLA test set (a folder of `.npy` files).
2. From the repo root, run:
   ```bash
   python run.py /path/to/kla_test_set restored_test_outputs_check
   ```
   (using a different output folder name here so you don't overwrite ours —
   feel free to compare the two afterward).
3. Confirm the file count matches:
   ```bash
   python -c "import os; print(len(os.listdir('restored_test_outputs_check')))"
   ```
4. (Optional) Sanity-check that outputs are well-formed — no manual
   inspection needed, this script checks shape/range/NaNs automatically:
   ```bash
   python -c "
import numpy as np, os
folder = 'restored_test_outputs_check'
for fname in sorted(os.listdir(folder))[:5]:
    arr = np.load(os.path.join(folder, fname))
    print(fname, 'shape=', arr.shape, 'dtype=', arr.dtype,
          'min=', arr.min(), 'max=', arr.max(),
          'has_nan_or_inf=', not np.isfinite(arr).all())
"
   ```
   Expected: `dtype=float32`, `min>=0.0`, `max<=1.0`, `has_nan_or_inf=False`,
   and shape exactly double the corresponding input file's shape.

---

## 6. Component 6 — `requirements.txt`

Pip freeze from our training environment — install with the single command
in Section 1. If you ever need to hand-rebuild this list, the minimum
required packages are: `torch`, `torchvision`, `numpy`, `pillow`,
`pytorch-msssim`.

---

## Troubleshooting

| Error you see | Why it happens | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'models'` | You ran the command from inside a different folder, not the repo root | `cd` into the folder that directly contains `run.py`, then re-run the command |
| `ERROR: weights file not found at 'models/restorenet_final.pt'` | The weights file is missing or you're not in the repo root | Make sure `models/restorenet_final.pt` exists; run commands from the repo root |
| `ERROR: no .npy files found in <input_dir>` | The input folder is empty, or the path is wrong, or the files aren't `.npy` | Double-check the folder path and that it actually contains `.npy` files |
| `pip: command not found` | Some systems only have `pip3` | Use `python3 -m pip install -r requirements.txt` instead |
| `CUDA out of memory` (during `train.py` only) | Batch size/crop size too big for your GPU | Add `--batch_size 8` or `--base_ch 48 --n_res 3` to lower memory use |
| Training is extremely slow | You're running `train.py` on a CPU | Expected — use a GPU (Colab/Kaggle) for real training runs; `run.py` (inference) is fine on CPU |

---

# 0) sanity-check the new model file compiles and passes self-tests
python models/model.py   # if placed at models/model.py run: python -m models.model ... or just:
                         # python -c "from models.model import RestoreNet, count_params; print(count_params(RestoreNet()))"

# 1) retrain (old weights are still loadable, but the v3 gains need retraining)
python train.py \
  --gt_dir /path/to/train/GT --lr_dir /path/to/train/NoisyLR \
  --out_dir checkpoints \
  --epochs_stage1 15 --epochs_stage2 80 --epochs_stage3 5 \
  --batch_size 16 --crop_lr 128 --base_ch 64 --n_res 4 --n_middle 8

# 2) promote best weights (start with best-by-SSIM EMA; also try best_psnr)
cp checkpoints/restorenet_best_ema.pt models/restorenet_final.pt

# 3) regenerate test outputs (TTA-8 on by default) and re-measure
python run.py <path_to_train_NoisyLR> restored_check_outputs
python evaluate.py --gt_dir /path/to/train/GT \
  --pred_dir restored_check_outputs --split_json checkpoints/test_split.json


## Architecture

`RestoreNet`: NAFNet-style encoder-decoder (SimpleGate + simplified channel attention + LayerNorm blocks) with PixelUnshuffle/PixelShuffle down/up-sampling, concat+1x1 skip fusion, a deep 8-block bottlencek (--n_middle), a pre_sr refinement block, an auxiliary LR denoise head (Stage-1 target, plus a small auxiliary loss during Stage 2), and a PixelShuffle super-resolution head with global residual learning on a bicubic-upsampled input. Training uses a combined Charbonnier + SSIM + VGG-perceptual + Sobel-edge + frequency-domain + range-penalty loss, per-stage cosine schedules, EMA weights, and an optional pixel-only polish stage. Inference uses an 8-way geometric self-ensemble (TTA). See models/model.py for the module-level design notes

---
Team: `PixelRevive`

