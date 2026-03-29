"""
Single-image inference with Deep3D (PyTorch version).

Given a 2D image (left eye), predict the right-eye view and save the result
as an anaglyph (red-cyan) or side-by-side stereo image.

Usage
-----
python infer.py --model exp/deep3d-0100.pth --input photo.jpg
                [--output output] [--format anaglyph|sbs|both]
                [--gpu 0]
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import Deep3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, scale=(-15, 17), upsample: int = 1,
               device='cpu') -> Deep3D:
    model = Deep3D(scale=scale, upsample=upsample).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(img_bgr: np.ndarray, data_shape=(384, 160),
               mean: np.ndarray = None):
    """Resize and normalise a BGR image.

    Returns
    -------
    left_t  : (1, 3, H, W) float32 tensor, mean-subtracted
    left0_t : (1, 3, H, W) float32 tensor, original values
    orig_h, orig_w : original image dimensions
    """
    orig_h, orig_w = img_bgr.shape[:2]
    W, H = data_shape
    img = cv2.resize(img_bgr, (W, H)).astype(np.float32)
    left0 = img.copy()
    if mean is not None:
        img -= mean[np.newaxis, np.newaxis, :]
    left_t  = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    left0_t = torch.from_numpy(left0.transpose(2, 0, 1)).unsqueeze(0)
    return left_t, left0_t, orig_h, orig_w


def postprocess(pred_t: torch.Tensor, orig_h: int, orig_w: int) -> np.ndarray:
    """Convert network output tensor to a BGR uint8 image."""
    pred = pred_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    if pred.shape[:2] != (orig_h, orig_w):
        pred = cv2.resize(pred, (orig_w, orig_h))
    return pred


def anaglyph(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Red-cyan anaglyph composite (left→red channel, right→cyan)."""
    out = np.zeros_like(left)
    out[:, :, :2] = right[:, :, :2]   # blue + green from right
    out[:, :, 2]  = left[:, :, 2]     # red from left
    return out


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Side-by-side stereo image."""
    H, W = left.shape[:2]
    half = W // 2
    out = np.zeros_like(left)
    out[:, :half]  = cv2.resize(left,  (half, H))
    out[:, half:]  = cv2.resize(right, (W - half, H))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description='Deep3D single-image inference')
    parser.add_argument('--model',   required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--input',   required=True,
                        help='Path to input 2D image')
    parser.add_argument('--output',  default='output',
                        help='Output file prefix (no extension)')
    parser.add_argument('--format',  default='both',
                        choices=['anaglyph', 'sbs', 'both'],
                        help='Output format')
    parser.add_argument('--mean',    default=None,
                        help='Path to mean.npy (optional)')
    parser.add_argument('--gpu',     type=int, default=0)
    return parser.parse_args()


def main():
    args = get_args()

    device = (torch.device(f'cuda:{args.gpu}')
              if args.gpu >= 0 and torch.cuda.is_available()
              else torch.device('cpu'))
    print(f'Using device: {device}')

    # Load model
    model = load_model(args.model, scale=(-15, 17), upsample=1, device=device)

    # Load mean
    mean = None
    if args.mean and os.path.exists(args.mean):
        mean = np.load(args.mean)
    elif os.path.exists(os.path.join(os.path.dirname(args.model), 'mean.npy')):
        mean = np.load(os.path.join(os.path.dirname(args.model), 'mean.npy'))

    # Load input image
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {args.input}')

    left_t, left0_t, orig_h, orig_w = preprocess(img, data_shape=(384, 160), mean=mean)
    left_t  = left_t.to(device)
    left0_t = left0_t.to(device)

    with torch.no_grad():
        pred_t = model(left_t, left0_t)

    right_img = postprocess(pred_t, orig_h, orig_w)
    left_img  = cv2.resize(img, (orig_w, orig_h))

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    if args.format in ('anaglyph', 'both'):
        out = anaglyph(left_img, right_img)
        path = args.output + '_anaglyph.jpg'
        cv2.imwrite(path, out)
        print(f'Saved anaglyph to {path}')

    if args.format in ('sbs', 'both'):
        out = side_by_side(left_img, right_img)
        path = args.output + '_sbs.jpg'
        cv2.imwrite(path, out)
        print(f'Saved side-by-side to {path}')

    # Also save the predicted right view
    cv2.imwrite(args.output + '_right.jpg', right_img)
    print(f'Saved right view to {args.output}_right.jpg')


if __name__ == '__main__':
    main()
