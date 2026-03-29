"""
Video / image parsing utilities for Deep3D data preparation.

No MXNet dependency – uses only OpenCV and NumPy.

Key functions
-------------
split(frame)           Split a side-by-side 3-D frame into left/right views.
anaglyph(left, right)  Compose a red-cyan anaglyph.
sbs(left, right)       Compose a side-by-side image.
flow(prev, cur)        Compute TV-L1 optical flow between two frames.
get_clip_rect(fname)   Auto-detect the active region of a 3-D video.
make_frame_data(...)   Extract left/right frame JPEG bytes into flat .npz files
                       or directly into LMDB (see data_prep.py).
make_flow_data(...)    Compute and store optical flow.
make_depth_data(...)   Compute SGBM disparity map.
process_movie(...)     Full pipeline for a side-by-side 3-D video.
process_movie2d(...)   Pipeline for a 2-D video (use same frame as both eyes).
"""

import argparse
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Frame splitting and compositing
# ---------------------------------------------------------------------------

def split(frame: np.ndarray, reshape=(256, 144), vert=True, clip=None):
    """Split a stereo frame into left and right views.

    Parameters
    ----------
    frame   : BGR HWC array
    reshape : (width, height) to resize each half to, or None to skip resize
    vert    : True  → side-by-side split (left | right)
              False → over-under split (top / bottom)
              None  → both halves are the full frame (2-D source)
    clip    : (x0, y0, x1, y1) crop applied *before* the split
    """
    if clip is not None:
        frame = frame[clip[1]:clip[3], clip[0]:clip[2]]

    if vert is True:
        mid = frame.shape[1] // 2
        lframe = frame[:, :mid]
        rframe = frame[:, mid:]
    elif vert is False:
        mid = frame.shape[0] // 2
        lframe = frame[:mid, :]
        rframe = frame[mid:, :]
    else:
        lframe = rframe = frame

    if reshape is not None:
        lframe = cv2.resize(lframe, reshape)
        rframe = cv2.resize(rframe, reshape)

    return lframe, rframe


def anaglyph(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return a red-cyan anaglyph: left→red, right→blue+green."""
    out = np.zeros_like(left)
    out[:, :, :2] = right[:, :, :2]
    out[:, :, 2]  = left[:, :, 2]
    return out


def sbs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return a side-by-side stereo composite."""
    H, W = left.shape[:2]
    half = W // 2
    out = np.zeros_like(left)
    out[:, :half]  = cv2.resize(left,  (half, H))
    out[:, half:]  = cv2.resize(right, (W - half, H))
    return out


# ---------------------------------------------------------------------------
# Optical flow
# ---------------------------------------------------------------------------

_tvl1 = None  # lazily initialised

def flow(prev_frame: np.ndarray, cur_frame: np.ndarray) -> np.ndarray:
    """Compute TV-L1 optical flow between two BGR frames.

    Returns a float32 (H, W, 2) array.
    """
    global _tvl1
    if _tvl1 is None:
        _tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    pg = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(cur_frame,  cv2.COLOR_BGR2GRAY)
    return _tvl1.calc(pg, cg, None)


# ---------------------------------------------------------------------------
# SGBM stereo depth
# ---------------------------------------------------------------------------

class Stereo:
    """Thin wrapper around OpenCV SGBM."""

    def __init__(self):
        self.stereo = None
        self._min   = 0
        self._scale = 16

    def sgbm_create(self, minDisparity=-15, numDisparities=32,
                    blockSize=16,
                    mode=cv2.StereoSGBM_MODE_HH):
        self._min   = minDisparity - 1
        self._scale = 16
        self.stereo = cv2.StereoSGBM_create(
            minDisparity   = minDisparity,
            numDisparities = numDisparities,
            blockSize      = blockSize,
            P1             = 8  * 3 * 5 ** 2,
            P2             = 32 * 3 * 5 ** 2,
            disp12MaxDiff  = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange   = 32,
            mode           = mode,
        )

    def compute(self, lframe: np.ndarray,
                rframe: np.ndarray) -> np.ndarray:
        lg = cv2.cvtColor(lframe, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)
        d  = self.stereo.compute(lg, rg)
        return (d / self._scale - self._min).astype(np.uint8)


# ---------------------------------------------------------------------------
# Frame extraction helpers (write raw JPEG bytes to disk for data_prep.py)
# ---------------------------------------------------------------------------

def _encode_jpg(img: np.ndarray, quality=80) -> bytes:
    """Encode a BGR image to JPEG bytes."""
    ok, buf = cv2.imencode('.jpg', img,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, "JPEG encoding failed"
    return buf.tobytes()


def make_frame_data(input_path: str, output_prefix: str,
                    reshape=(256, 144), vert=True, clip=None,
                    quality: int = 80):
    """Extract frame pairs from a stereo video and save as JPEG byte lists.

    Output files
    ------------
    <output_prefix>_l.bin  – packed left-eye JPEG bytes
    <output_prefix>_r.bin  – packed right-eye JPEG bytes

    Each .bin file is a NumPy object array of bytes objects, suitable for
    direct insertion into LMDB via ``data_prep.add_to_db``.

    Parameters
    ----------
    input_path    : path to the source video
    output_prefix : path prefix for output files
    reshape       : target (width, height) for each half frame
    vert          : side-by-side (True) or over-under (False) split
    clip          : (x0, y0, x1, y1) crop before split
    quality       : JPEG quality (0–100)
    """
    cap = cv2.VideoCapture(input_path)
    l_frames, r_frames = [], []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % 5000 == 0:
            print(f'frame {i}')
        lf, rf = split(frame, reshape=reshape, vert=vert, clip=clip)
        l_frames.append(_encode_jpg(lf, quality))
        r_frames.append(_encode_jpg(rf, quality))
        i += 1
    cap.release()

    np.save(output_prefix + '_l.npy', np.array(l_frames, dtype=object))
    np.save(output_prefix + '_r.npy', np.array(r_frames, dtype=object))
    print(f'Saved {i} frame pairs to {output_prefix}_[l|r].npy')
    return l_frames, r_frames


def make_flow_data(l_frames: list, output_prefix: str, quality: int = 80):
    """Compute TV-L1 optical flow for each consecutive left-frame pair.

    Saves an object array of JPEG-encoded 3-channel flow images to
    ``<output_prefix>_flow.npy``.
    """
    flow_frames = []
    pframe = None
    for i, buf in enumerate(l_frames):
        arr   = np.frombuffer(buf, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if pframe is None:
            pframe = frame
        f = flow(pframe, frame)
        # Normalise to [0, 255] uint8 (same as original: clip((f+40)*255/80))
        f_vis = np.clip((f + 40) * 255.0 / 80.0, 0, 255).astype(np.uint8)
        f_3ch = np.concatenate(
            [f_vis, np.zeros((f_vis.shape[0], f_vis.shape[1], 1), dtype=np.uint8)],
            axis=2
        )
        pframe = frame
        flow_frames.append(_encode_jpg(f_3ch, quality))
        if i % 10 == 0:
            print(f'flow {i}')

    np.save(output_prefix + '_flow.npy', np.array(flow_frames, dtype=object))
    print(f'Saved {len(flow_frames)} flow frames to {output_prefix}_flow.npy')
    return flow_frames


def make_depth_data(l_frames: list, r_frames: list,
                    output_prefix: str):
    """Compute SGBM disparity between left/right frame pairs.

    Saves depth PNG bytes to ``<output_prefix>_depth.npy``.
    """
    stereo = Stereo()
    stereo.sgbm_create(minDisparity=-15, numDisparities=32, blockSize=16)
    depth_frames = []
    for i, (lb, rb) in enumerate(zip(l_frames, r_frames)):
        la = np.frombuffer(lb, dtype=np.uint8)
        ra = np.frombuffer(rb, dtype=np.uint8)
        lf = cv2.imdecode(la, cv2.IMREAD_COLOR)
        rf = cv2.imdecode(ra, cv2.IMREAD_COLOR)
        d  = stereo.compute(rf, lf)
        ok, buf = cv2.imencode('.png', d,
                               [cv2.IMWRITE_PNG_COMPRESSION, 3])
        assert ok
        depth_frames.append(buf.tobytes())
        if i % 100 == 0:
            print(f'depth {i}')

    np.save(output_prefix + '_depth.npy',
            np.array(depth_frames, dtype=object))
    print(f'Saved {len(depth_frames)} depth frames to {output_prefix}_depth.npy')
    return depth_frames


# ---------------------------------------------------------------------------
# Clip detection
# ---------------------------------------------------------------------------

def get_clip_rect(fname: str, vert=True):
    """Auto-detect the active (non-black) region of a 3-D video.

    Returns (x0, y0, x1, y1) suitable for passing to ``split`` / ``make_frame_data``.
    """
    assert vert, "Only vertical (side-by-side) format is supported."
    cap = cv2.VideoCapture(fname)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    skip_frames = fps * 60 * 2  # skip first 2 minutes of logos
    accum_frames = fps * 60     # accumulate 1 minute of frames
    # Skip first 2 minutes of logos
    for _ in range(skip_frames):
        assert cap.isOpened()
        assert cap.read()[0]

    shape = cap.read()[1].shape
    print(f'Original shape: {shape}')
    assert shape[1] == 1920, f'Expected 1920-wide video, got {shape[1]}'
    assert shape[0] >= 800

    acc = np.zeros(shape, dtype=np.float64)
    for _ in range(accum_frames):
        ret, frame = cap.read()
        assert ret
        acc += frame
    acc /= accum_frames

    y0 = 0
    while acc[y0].mean() < 2:
        y0 += 1
    y1 = shape[0] - 1
    while acc[y1].mean() < 2:
        y1 -= 1
    y1 += 1

    print(f'Clip height: {y1 - y0}, width: {shape[1]}')
    diff = (y1 - y0) - 800
    half = diff // 2
    y0 += half
    y0 = max(y0, 0)
    diff = (y1 - y0) - 800
    y1 -= diff
    y1 = min(y1, shape[0])
    cap.release()
    return (0, y0, 1920 // 2, y1)


# ---------------------------------------------------------------------------
# High-level pipelines
# ---------------------------------------------------------------------------

def process_movie(fname: str, prefix: str, reshape=(432, 180), vert=True):
    """Full 3-D video extraction pipeline."""
    print(f'Processing 3D video {fname} → {prefix}')
    rect = get_clip_rect(fname, vert)
    print(f'Clipping to {rect}')
    make_frame_data(fname, prefix, reshape=reshape, vert=vert, clip=rect)
    # Uncomment to also compute flow / depth:
    # l_frames = list(np.load(prefix + '_l.npy', allow_pickle=True))
    # r_frames = list(np.load(prefix + '_r.npy', allow_pickle=True))
    # make_flow_data(l_frames, prefix)
    # make_depth_data(l_frames, r_frames, prefix)


def process_movie2d(fname: str, prefix: str, reshape=(432, 180)):
    """2-D video extraction (uses full frame as both eyes)."""
    print(f'Processing 2D video {fname} → {prefix}')
    rect = list(get_clip_rect(fname, True))
    rect[2] = rect[0] + (rect[2] - rect[0]) * 2
    print(f'Clipping to {rect}')
    make_frame_data(fname, prefix, reshape=reshape, vert=None, clip=rect)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse 3D/2D video into per-frame JPEG byte arrays'
    )
    parser.add_argument('path',   type=str, help='Path to video file')
    parser.add_argument('prefix', type=str, help='Output file prefix')
    parser.add_argument('--sbs3d', action='store_true', default=True,
                        help='Input is a side-by-side 3-D video (default)')
    parser.add_argument('--2d',   dest='is_2d', action='store_true',
                        help='Input is a plain 2-D video')
    args = parser.parse_args()

    if args.is_2d:
        process_movie2d(args.path, args.prefix)
    else:
        process_movie(args.path, args.prefix)
