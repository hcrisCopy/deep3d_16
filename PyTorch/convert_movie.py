"""
Batch video-to-3D conversion with Deep3D (PyTorch version).

Reads frames from a source video (or from the LMDB test set) and writes
output MKV files:
  <prefix>_l.mkv         – left-eye input frames
  <prefix>_ana_p.mkv     – anaglyph (left + predicted right)
  <prefix>_sbs_p.mkv     – side-by-side (left | predicted right)

Usage
-----
python convert_movie.py --model exp/deep3d-0100.pth --input movie.mp4
                        [--output output] [--gpu 0] [--n-batch 50]
                        [--batch-size 10]
"""

import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import Deep3D
from infer import load_model, preprocess, postprocess, anaglyph, side_by_side


def get_args():
    parser = argparse.ArgumentParser(description='Convert 2D video to 3D')
    parser.add_argument('--model',      required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--input',      required=True,
                        help='Path to input video file')
    parser.add_argument('--output',     default='output',
                        help='Output file prefix (no extension)')
    parser.add_argument('--gpu',        type=int, default=0)
    parser.add_argument('--n-batch',    type=int, default=50,
                        help='Number of batches to process')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--mean',       default=None,
                        help='Path to mean.npy')
    return parser.parse_args()


def process_video(model: Deep3D, cap: cv2.VideoCapture,
                  mean: np.ndarray, device,
                  data_shape=(384, 160),
                  n_batch: int = 50, batch_size: int = 10,
                  output_prefix: str = 'output'):
    """Read *batch_size* × *n_batch* frames and write stereo output videos."""

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (orig_w, orig_h)

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    lcap     = cv2.VideoWriter(output_prefix + '_l.mkv',     fourcc, fps, out_size)
    pcap_ana = cv2.VideoWriter(output_prefix + '_ana_p.mkv', fourcc, fps, out_size)
    pcap_sbs = cv2.VideoWriter(output_prefix + '_sbs_p.mkv', fourcc, fps, out_size)

    W, H = data_shape
    total_mae = 0.0
    n_frames  = 0

    model.eval()
    with torch.no_grad():
        for b in range(n_batch):
            # Read a mini-batch of frames
            frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            if not frames:
                break

            # Preprocess batch
            left_list  = []
            left0_list = []
            for fr in frames:
                lt, l0t, _, _ = preprocess(fr, data_shape=data_shape, mean=mean)
                left_list.append(lt)
                left0_list.append(l0t)

            left_batch  = torch.cat(left_list,  dim=0).to(device)
            left0_batch = torch.cat(left0_list, dim=0).to(device)

            pred_batch = model(left_batch, left0_batch)

            # Write frames
            for i, (fr, pred_t) in enumerate(zip(frames, pred_batch)):
                right_img = postprocess(pred_t.unsqueeze(0), orig_h, orig_w)
                left_img  = cv2.resize(fr, (orig_w, orig_h))

                lcap.write(left_img)
                pcap_ana.write(anaglyph(left_img, right_img))
                pcap_sbs.write(side_by_side(left_img, right_img))

                # MAE (just for logging)
                mae = np.abs(left_img.astype(np.float32) -
                             right_img.astype(np.float32)).mean()
                total_mae += mae
                n_frames  += 1

            print(f'Batch {b+1}/{n_batch} done – '
                  f'avg MAE so far: {total_mae/n_frames:.2f}')

    lcap.release()
    pcap_ana.release()
    pcap_sbs.release()
    print(f'Output written to {output_prefix}_[l|ana_p|sbs_p].mkv')


def main():
    args = get_args()

    device = (torch.device(f'cuda:{args.gpu}')
              if args.gpu >= 0 and torch.cuda.is_available()
              else torch.device('cpu'))
    print(f'Using device: {device}')

    model = load_model(args.model, scale=(-15, 17), upsample=1, device=device)

    mean = None
    if args.mean and os.path.exists(args.mean):
        mean = np.load(args.mean)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {args.input}')

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    process_video(model, cap, mean, device,
                  data_shape=(384, 160),
                  n_batch=args.n_batch,
                  batch_size=args.batch_size,
                  output_prefix=args.output)
    cap.release()


if __name__ == '__main__':
    main()
