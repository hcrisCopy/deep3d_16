"""
PyTorch Dataset for Deep3D stereo training data stored in LMDB.

Data format (created by data_prep.py / parse.py)
-------------------------------------------------
An LMDB database at ``db_path`` with two named sub-databases:
  'l'  – left-eye JPEG frames
  'r'  – right-eye JPEG frames

Keys are zero-padded 9-digit decimal strings: ``b'%09d' % idx``.

Metadata stored in the **default** (unnamed) sub-database:
  'N'                  – number of movies (int as ASCII string)
  'prefix_list'        – comma-separated movie names
  'chunk_size'         – frames per chunk
  'chunk_base'         – chunk addressing offset
  'movie_base'         – per-movie addressing offset
  'train_idx'          – comma-separated sample indices for training
  'valid_idx'          – comma-separated sample indices for validation
  'test_idx'           – comma-separated sample indices for testing
  'shuffled_test_idx'  – shuffled test indices

Image values
  Images are stored as raw JPEG bytes produced by
  ``cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])``.
"""

import random
import io
import lmdb
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _decode_jpg(buf: bytes) -> np.ndarray:
    """Decode JPEG bytes → BGR uint8 HWC array."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None, "Failed to decode JPEG image"
    return img


def _crop(img: np.ndarray, p, shape, margin=0, test=False, grid=1):
    """Randomly (or deterministically) crop *img* to *shape* (W, H).

    Returns the cropped image and the crop origin *p* (col_grid, row_grid).
    """
    if p is None:
        max_col = (img.shape[1] - shape[0] - margin) // grid
        max_row = (img.shape[0] - shape[1]) // grid
        if test:
            p = (max_col // 2, max_row // 2)
        else:
            p = (random.randint(0, max_col), random.randint(0, max_row))
    col = p[0] * grid
    row = p[1] * grid
    return img[row:row + shape[1], col:col + shape[0]], p


class Mov3dDataset(Dataset):
    """PyTorch Dataset that reads stereo frame pairs from an LMDB database.

    Parameters
    ----------
    db_path : str
        Path to the LMDB directory created by ``data_prep.py``.
    data_shape : tuple(int, int)
        (width, height) of the network input (e.g. (384, 160)).
    scale : tuple(int, int)
        Disparity range, used only to compute the crop margin when
        *output_depth* is True.
    split : str
        One of ``'train'``, ``'valid'``, ``'test'``.
    mean : np.ndarray or None
        Per-channel BGR mean with shape (3,).  If None, use zeros.
    test_mode : bool
        When True disable random crop and shuffling.
    output_depth : bool
        When True also return a depth pseudo-label (not used in the default
        training pipeline).
    stride : int
        Frame stride when stacking multiple frames (currently unused;
        data_frames is fixed at 1 matching the default training config).
    """

    def __init__(self, db_path: str, data_shape=(384, 160),
                 scale=(-15, 17), split='train',
                 mean=None, test_mode=False, output_depth=False,
                 stride: int = 1):
        super().__init__()
        self.data_shape = data_shape   # (W, H)
        self.scale = scale
        self.test_mode = test_mode
        self.output_depth = output_depth
        self.stride = stride
        self.margin = (scale[1] - scale[0]) // 2 if output_depth else 0

        # Mean (BGR, float32, shape=(3,))
        if mean is None:
            self.mean = np.zeros(3, dtype=np.float32)
        else:
            self.mean = np.array(mean, dtype=np.float32)

        # Open LMDB (read-only)
        self.env = lmdb.open(db_path, map_size=1 << 40, max_dbs=5,
                             readonly=True, readahead=False, lock=False)
        self.ldb = self.env.open_db(b'l')
        self.rdb = self.env.open_db(b'r')

        # Load index list for requested split
        with self.env.begin() as txn:
            key = {
                'train': b'train_idx',
                'valid': b'valid_idx',
                'test':  b'shuffled_test_idx',
            }[split]
            raw = txn.get(key)
            if raw is None:
                raise KeyError(
                    f"Key '{key.decode()}' not found in LMDB.  "
                    "Run data_prep.py first."
                )
            self.idx = [int(v) for v in raw.split(b',') if v]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        """Return (left_tensor, left0_tensor, right_tensor).

        Shapes
        ------
        left_tensor  : (3, H, W)  float32, mean-subtracted
        left0_tensor : (3, H, W)  float32, original scale (copy of left)
        right_tensor : (3, H, W)  float32, original scale
        """
        idx = self.idx[i]
        W, H = self.data_shape

        with self.env.begin() as txn:
            sl = txn.get(('%09d' % idx).encode(), db=self.ldb)
            sr = txn.get(('%09d' % idx).encode(), db=self.rdb)

        if sl is None or sr is None:
            # Fall back to a random sample if this one is missing
            return self.__getitem__(random.randint(0, len(self) - 1))

        limg = _decode_jpg(sl)   # BGR uint8 HWC
        rimg = _decode_jpg(sr)

        # Resize to data_shape if necessary
        if limg.shape[:2] != (H, W):
            limg = cv2.resize(limg, (W, H))
        if rimg.shape[:2] != (H, W):
            rimg = cv2.resize(rimg, (W, H))

        # Random crop
        limg, p = _crop(limg, None, (W, H), self.margin, test=self.test_mode)
        rimg, _ = _crop(rimg, p,    (W, H), 0,           test=self.test_mode)

        # float32, channel-last → channel-first
        limg = limg.astype(np.float32)
        rimg = rimg.astype(np.float32)

        # left0: copy before mean subtraction (used for DepthDot)
        left0 = limg.copy()

        # Subtract per-channel mean from left (BGR order)
        limg -= self.mean[np.newaxis, np.newaxis, :]

        # HWC → CHW
        left_t  = torch.from_numpy(limg.transpose(2, 0, 1))
        left0_t = torch.from_numpy(left0.transpose(2, 0, 1))
        right_t = torch.from_numpy(rimg.transpose(2, 0, 1))

        return left_t, left0_t, right_t


def compute_mean(db_path: str, data_shape=(384, 160),
                 n_samples: int = 5000) -> np.ndarray:
    """Estimate the per-channel BGR mean from *n_samples* training frames."""
    env = lmdb.open(db_path, map_size=1 << 40, max_dbs=5,
                    readonly=True, readahead=False, lock=False)
    ldb = env.open_db(b'l')
    W, H = data_shape

    with env.begin() as txn:
        raw = txn.get(b'train_idx')
        idx_list = [int(v) for v in raw.split(b',') if v]

    random.shuffle(idx_list)
    idx_list = idx_list[:n_samples]

    acc = np.zeros(3, dtype=np.float64)
    count = 0
    with env.begin() as txn:
        for idx in idx_list:
            buf = txn.get(('%09d' % idx).encode(), db=ldb)
            if buf is None:
                continue
            img = _decode_jpg(buf)
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H))
            acc += img.reshape(-1, 3).mean(axis=0)
            count += 1

    mean = (acc / count).astype(np.float32) if count > 0 else np.zeros(3, np.float32)
    env.close()
    return mean
