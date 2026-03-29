# Deep3D – PyTorch Implementation

Automatic 2D-to-3D video conversion using deep convolutional neural networks,
re-implemented from scratch in **PyTorch** (fully independent of MXNet).

> **Original paper**: *Deep3D: Fully Automatic 2D-to-3D Video Conversion with
> Deep Convolutional Neural Networks*, Junyuan Xie et al., ECCV 2016.

---

## Table of Contents

1. [Method Overview](#1-method-overview)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Preparing Training Data](#4-preparing-training-data)
   - 4.1 [Prerequisites – 3D video files](#41-prerequisites--3d-video-files)
   - 4.2 [Extract frames from videos](#42-extract-frames-from-videos)
   - 4.3 [Build the LMDB database](#43-build-the-lmdb-database)
   - 4.4 [Split into train / valid / test](#44-split-into-train--valid--test)
5. [Training](#5-training)
   - 5.1 [Quick start](#51-quick-start)
   - 5.2 [Command-line options](#52-command-line-options)
   - 5.3 [Monitoring](#53-monitoring)
   - 5.4 [Checkpoints](#54-checkpoints)
6. [Inference](#6-inference)
   - 6.1 [Single image](#61-single-image)
   - 6.2 [Video conversion](#62-video-conversion)
7. [Network Architecture](#7-network-architecture)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Method Overview

Deep3D converts a 2D (left-eye) image into a stereo pair by learning a
**per-pixel disparity distribution** rather than a single depth value.  
The network outputs a soft probability map over 33 disparity levels
(−15 … +17 pixels).  A custom **DepthDot** layer then reconstructs the right-eye
view as a weighted sum of horizontally-shifted copies of the input image,
making the whole pipeline end-to-end trainable with an L1 loss against real
right-eye frames extracted from 3D movies.

During inference only the left-eye image is required as input.

---

## 2. Project Structure

```
PyTorch/
├── model.py          # Deep3D nn.Module, DepthDot autograd Function, MAE loss
├── dataset.py        # PyTorch Dataset reading from LMDB
├── train.py          # Training script
├── infer.py          # Single-image inference
├── convert_movie.py  # Batch video-to-3D conversion
├── parse.py          # Data preparation utilities (frame extraction, flow, depth)
├── data_prep.py      # LMDB database builder & index management
└── README.md         # This file
```

All scripts are self-contained and have no dependency on the parent project or
on MXNet.

---

## 3. Environment Setup

### Requirements

| Package        | Minimum version | Notes                                  |
|----------------|-----------------|----------------------------------------|
| Python         | 3.8             |                                        |
| PyTorch        | 2.0             | with CUDA if GPU training is desired   |
| torchvision    | 0.15            | needed for VGG-16 pretrained weights   |
| OpenCV         | 4.5             | `opencv-python` or `opencv-contrib-python` |
| lmdb           | 1.3             |                                        |
| NumPy          | 1.21            |                                        |

### Installation (conda)

```bash
conda create -n deep3d python=3.10 -y
conda activate deep3d

# PyTorch with CUDA 11.8 – adjust for your CUDA version
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install opencv-contrib-python lmdb numpy
```

### Installation (pip + venv)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-contrib-python lmdb numpy
```

> **CPU-only machines**: replace the CUDA variant of PyTorch with the CPU
> wheel.  Training will be very slow; inference on a modern CPU is feasible for
> single images.

---

## 4. Preparing Training Data

### 4.1 Prerequisites – 3D video files

Deep3D is trained on **side-by-side (SBS) 3D movies**, where the left half of
each frame is the left-eye view and the right half is the right-eye view.
Typical source material:

* Commercial 3D Blu-rays ripped as MKV/MP4 with full-SBS encoding
  (1920 × 1080, each eye 960 × 1080)
* 3D trailers or YouTube "Half-SBS" videos

Place raw video files anywhere accessible, e.g.:

```
data/
  raw/
    movie_a.mkv
    movie_b.mkv
```

### 4.2 Extract frames from videos

```bash
cd PyTorch/

# For each SBS 3D video:
python parse.py data/raw/movie_a.mkv data/raw/movie_a
python parse.py data/raw/movie_b.mkv data/raw/movie_b
# ...
```

This produces two NumPy object-array files per movie:
```
data/raw/movie_a_l.npy   # left-eye JPEG bytes, one entry per frame
data/raw/movie_a_r.npy   # right-eye JPEG bytes
```

Each frame is resized to **432 × 180** pixels (configurable via `--reshape`).

**For a plain 2D video** (the network will use the same frame as both eyes
during pre-training or domain adaptation):

```bash
python parse.py data/raw/some_2d.mp4 data/raw/some_2d --2d
```

**Optional extras** (optical flow / depth, uncomment the relevant lines in
`parse.py`):

```bash
# Compute TV-L1 optical flow for the left-eye sequence:
python - <<'EOF'
import numpy as np
from parse import make_flow_data
l = list(np.load('data/raw/movie_a_l.npy', allow_pickle=True))
make_flow_data(l, 'data/raw/movie_a')
EOF

# Compute SGBM stereo depth:
python - <<'EOF'
import numpy as np
from parse import make_depth_data
l = list(np.load('data/raw/movie_a_l.npy', allow_pickle=True))
r = list(np.load('data/raw/movie_a_r.npy', allow_pickle=True))
make_depth_data(l, r, 'data/raw/movie_a')
EOF
```

### 4.3 Build the LMDB database

```bash
python data_prep.py add data/raw data/lmdb
```

This scans `data/raw/` for all `*_l.npy` files, and inserts each movie into the
LMDB at `data/lmdb/`.  Both `data/lmdb/` and `data/raw/` are created
automatically if they do not exist.

### 4.4 Split into train / valid / test

```bash
python data_prep.py shuffle data/lmdb
```

The default split reserves 30 % of **movies** as the test set and 10 % of the
remaining training frames as the validation set.  The exact proportions can be
adjusted:

```bash
python data_prep.py shuffle data/lmdb --valid-ratio 0.05 --test-ratio 0.2
```

**Verify integrity:**

```bash
python data_prep.py show data/lmdb
```

Expected output (numbers depend on your data):
```
Movies: 8  chunk_size: 4320  chunk_base: 10000  movie_base: 1000000
Prefixes: ['movie_a', 'movie_b', ...]
train_idx: 94251 samples
valid_idx: 10472 samples
shuffled_test_idx: 41053 samples
```

---

## 5. Training

### 5.1 Quick start

```bash
cd PyTorch/

# GPU training (GPU 0), VGG-16 pretrained encoder, default hyperparameters:
python train.py \
    --data     data/lmdb \
    --output   exp/deep3d \
    --gpu      0 \
    --pretrained
```

### 5.2 Command-line options

| Argument         | Default         | Description                                         |
|------------------|-----------------|-----------------------------------------------------|
| `--data`         | `data/lmdb`     | Path to the LMDB directory                          |
| `--batch-size`   | `64`            | Training mini-batch size                            |
| `--epochs`       | `100`           | Total training epochs                               |
| `--lr`           | `2e-3`          | Initial learning rate (SGD)                         |
| `--lr-step`      | `20`            | Decay LR by `--lr-gamma` every this many epochs     |
| `--lr-gamma`     | `0.1`           | LR decay factor                                     |
| `--wd`           | `0.0`           | Weight decay                                        |
| `--momentum`     | `0.9`           | SGD momentum                                        |
| `--output`       | `exp/deep3d`    | Checkpoint file prefix                              |
| `--gpu`          | `0`             | GPU id (`-1` for CPU)                               |
| `--pretrained`   | *(flag)*        | Load VGG-16 ImageNet weights into encoder           |
| `--resume`       | `None`          | Path to `.pth` checkpoint to resume from            |
| `--num-workers`  | `4`             | DataLoader worker processes                         |
| `--log-every`    | `10`            | Print loss every N batches                          |
| `--val-batches`  | `100`           | Validation batches evaluated per epoch              |

### 5.3 Monitoring

Training logs are written to both the console and a log file:
```
exp/deep3d_YYYY_MM_DD-HH_MM.log
```

Each log line looks like:
```
2024-01-15 12:34:56 INFO Epoch [3/100] Batch [100/1473] Loss: 8.4312 LR: 0.002000
2024-01-15 12:36:10 INFO Epoch [3/100] Train MAE: 8.2191  Val MAE: 9.0548
```

### 5.4 Checkpoints

A checkpoint is saved at the end of every epoch:
```
exp/deep3d-0001.pth
exp/deep3d-0002.pth
...
```

Each file is a dictionary with keys `epoch`, `model`, and `optimizer`.

**Resume training from epoch 50:**
```bash
python train.py --resume exp/deep3d-0050.pth --gpu 0
```

---

## 6. Inference

### 6.1 Single image

```bash
python infer.py \
    --model  exp/deep3d-0100.pth \
    --input  photo.jpg \
    --output results/photo \
    --format both           # 'anaglyph', 'sbs', or 'both'
    --gpu 0
```

Output files:
```
results/photo_right.jpg      # predicted right-eye view
results/photo_anaglyph.jpg   # red-cyan anaglyph
results/photo_sbs.jpg        # side-by-side composite
```

A `mean.npy` file (channel mean) is automatically loaded from the same
directory as the checkpoint if present.  You can also specify it explicitly:

```bash
python infer.py --model exp/deep3d-0100.pth --input photo.jpg \
                --mean  data/lmdb/mean.npy
```

### 6.2 Video conversion

```bash
python convert_movie.py \
    --model      exp/deep3d-0100.pth \
    --input      myvideo.mp4 \
    --output     results/myvideo \
    --batch-size 10 \
    --n-batch    200 \
    --gpu        0
```

Output files:
```
results/myvideo_l.mkv       # left-eye (input) frames
results/myvideo_ana_p.mkv   # anaglyph (left + predicted right)
results/myvideo_sbs_p.mkv   # side-by-side stereo
```

| Argument       | Default    | Description                             |
|----------------|------------|-----------------------------------------|
| `--model`      | required   | Path to `.pth` checkpoint               |
| `--input`      | required   | Path to input video                     |
| `--output`     | `output`   | Output file prefix                      |
| `--gpu`        | `0`        | GPU id                                  |
| `--n-batch`    | `50`       | Number of batches to process            |
| `--batch-size` | `10`       | Frames per batch                        |
| `--mean`       | auto-detect| Path to `mean.npy`                      |

---

## 7. Network Architecture

```
Input: left-eye image  (B, 3, 160, 384)
                │
    ┌───────────┴────────────────────────────────────────┐
    │                  Encoder (VGG-style)               │
    │  conv1_1 → relu → pool1   (B,  64,  80, 192)       │
    │  conv2_1 → relu → pool2   (B, 128,  40,  96)       │
    │  conv3_1+3_2 → relu → pool3 (B,256, 20,  48)       │
    │  conv4_1+4_2 → relu → pool4 (B,512, 10,  24)       │
    │  conv5_1+5_2 → relu → pool5 (B,512,  5,  12)       │
    │  FC-6 (512) → FC-7 (512) → FC-8 (33×5×12)          │
    └─────────────────────────────────────────────────────┘
         │         │         │         │         │
       pred1     pred2     pred3     pred4     pred5
    BN+Conv     BN+Conv   BN+Conv   BN+Conv    reshape
     +deconv    +deconv   +deconv   +deconv   +deconv
    stride-1   stride-2  stride-4  stride-8  stride-16
         │         │         │         │         │
         └────────────────────┬────────────────────┘
                         ElementWiseSum
                              │
                           ReLU
                           Deconv ×2 + Conv
                              │
                      (B, 33, 160, 384)
                              │
                         Softmax (dim=1)
                              │
                         DepthDot (left0)
                              │
              Reconstructed right view (B, 3, 160, 384)
                              │
                         L1 Loss ← real right frame
```

**DepthDot** computes:

```
out[b, c, h, w] = Σ_{j=−15}^{+16}  softmax[b, j+16, h, w]
                                   × left0[b, c, h, clamp(w−j, 0, W−1)]
```

This shifts the left-eye image by each disparity amount and weights the result
by the predicted probability, producing a soft-rendered right-eye view.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `KeyError: 'train_idx'` | LMDB exists but `shuffle` was not run | Run `python data_prep.py shuffle data/lmdb` |
| `AssertionError: Failed to decode JPEG` | Corrupted frame in LMDB | Re-run `data_prep.py add` after verifying `.npy` files |
| CUDA OOM during training | Batch size too large | Reduce `--batch-size` |
| Black / near-black predicted right view | LR too high at the start | Use `--pretrained` and/or lower `--lr` to `5e-4` |
| `module 'cv2' has no attribute 'optflow'` | Need contrib build | `pip install opencv-contrib-python` |
| `lmdb.Error: MDB_NOTFOUND` | Sub-database not found | Ensure `data_prep.py add` completed without errors |
| Very slow DataLoader on Windows | `num_workers > 0` causes multiprocessing issues | Set `--num-workers 0` |
