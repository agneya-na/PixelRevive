# AI-Based Restoration of Degraded Images — Team PixelRevive

PixelRevive: deep learning model for restoring degraded grayscale semiconductor
inspection images with 2x resolution enhancement (denoising + super-resolution
in one forward pass).

## Folder structure

```
PixelRevive/
├── run.py              entry point (see Usage below)
├── requirements.txt
├── README.md
└── models/
    ├── model.py         RestoreNet architecture
    └── restorenet_final.pt   trained weights
```

## Setup

```bash
pip install -r requirements.txt
```

No other setup is required. `run.py` does not access the internet, does not
require API keys, and does not download anything at run time — the trained
weights are already included in `models/restorenet_final.pt`.

## Usage

```bash
python run.py <input_dir> <output_dir>
```

- `<input_dir>`: folder containing degraded input images as `.npy` files
  (2D grayscale arrays, values roughly in `[0, 1]` and may exceed 1.0 due to
  speckle noise — this is expected and handled).
- `<output_dir>`: created automatically if it doesn't exist.

For every `<name>.npy` in `<input_dir>`, `run.py` writes a restored
`<name>.npy` to `<output_dir>` — same filename, 2D array of shape `(H, W)`
at 2x the input resolution, float32 values clipped to `[0, 1]` with no
NaN/Inf.

Example:

```bash
python run.py test_data/NoisyLR restored_outputs
```

Runs on GPU automatically if available (`torch.cuda.is_available()`), falls
back to CPU otherwise.

## Model

Encoder-decoder with residual blocks and a PixelShuffle super-resolution
head, trained in two stages (denoise pretraining, then joint fine-tuning)
with a combined L1 + SSIM + perceptual + range-clipping loss. See
`models/model.py` for the architecture.
