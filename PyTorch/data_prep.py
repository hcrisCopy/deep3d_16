"""
Build and manage the LMDB training database for Deep3D.

Data format
-----------
An LMDB at ``db_path`` containing two named sub-databases (max_dbs=5):
  'l'  – left-eye frames (raw JPEG bytes)
  'r'  – right-eye frames (raw JPEG bytes)

Keys: ``b'%09d' % global_index``  (9-digit decimal, zero-padded)

Global index layout
  global_index = movie_id * movie_base
               + chunk_id * chunk_base
               + frame_within_chunk

Metadata in the default database (string keys, string values):
  'N'               – number of movies inserted
  'prefix_list'     – comma-separated list of movie names
  'chunk_size'      – max frames per chunk
  'chunk_base'      – chunk addressing stride
  'movie_base'      – per-movie base offset
  'idx'             – full chunk index ('$'-separated movies,
                      '|'-separated chunks, ','-separated frame indices)
  'train_idx'       – comma-separated train indices
  'valid_idx'       – comma-separated validation indices
  'test_idx'        – comma-separated test indices
  'shuffled_test_idx' – shuffled test indices

Usage
-----
1. Extract frame JPEG arrays from videos with parse.py:
       python parse.py /path/to/movie.mkv data/raw/mymovie

2. Add all movies to LMDB:
       python data_prep.py add data/raw data/lmdb

3. Shuffle and split:
       python data_prep.py shuffle data/lmdb

4. (Optional) verify integrity:
       python data_prep.py show data/lmdb
"""

import argparse
import os
import random

import lmdb
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_prop(txn, key: bytes, default, check=False):
    """Read a metadata value from LMDB, inserting the default if absent."""
    if isinstance(key, str):
        key = key.encode()
    v = txn.get(key)
    if v is None:
        txn.put(key, str(default).encode())
        return default
    v = type(default)(v.decode())
    if check:
        assert v == default, f'{v!r} != {default!r}'
    return v


# ---------------------------------------------------------------------------
# add_to_db
# ---------------------------------------------------------------------------

def add_to_db(db_path: str, npy_prefix: str,
              suffix_list=('l', 'r'),
              chunk_size=24 * 60 * 3,
              chunk_base=10_000,
              movie_base=1_000_000):
    """Insert one movie's frame arrays into the LMDB database.

    Parameters
    ----------
    db_path     : path to the LMDB directory
    npy_prefix  : prefix used when saving .npy files, e.g. ``data/raw/mymovie``
                  → reads ``mymovie_l.npy`` and ``mymovie_r.npy``
    suffix_list : which sub-databases to populate (default: ('l', 'r'))
    chunk_size  : frames per addressing chunk
    chunk_base  : addressing stride per chunk
    movie_base  : addressing stride per movie
    """
    prefix = os.path.basename(npy_prefix)

    env = lmdb.open(db_path, map_size=1 << 40, max_dbs=5)

    with env.begin(write=True) as txn:
        N          = _get_prop(txn, b'N',          0)
        chunk_size = _get_prop(txn, b'chunk_size', chunk_size, check=N > 0)
        chunk_base = _get_prop(txn, b'chunk_base', chunk_base, check=N > 0)
        movie_base = _get_prop(txn, b'movie_base', movie_base, check=N > 0)

        raw_plist = txn.get(b'prefix_list', b'')
        prefix_list = [p for p in raw_plist.decode().split(',') if p]
        if prefix in prefix_list:
            print(f'Prefix {prefix!r} already in database – skipping.')
            return

        raw_idx = txn.get(b'idx', b'')
        idx_str = raw_idx.decode() if raw_idx else ''
        idx = [i for i in idx_str.split('$') if i]

    prev_ichunk = prev_count = None

    for isuffix, suffix in enumerate(suffix_list):
        frames = list(np.load(f'{npy_prefix}_{suffix}.npy', allow_pickle=True))
        db   = env.open_db(suffix.encode())
        base = N * movie_base
        chunks = [[]]
        ichunk = 0
        count  = 0

        with env.begin(write=True) as txn:
            for frame_bytes in frames:
                if count == chunk_size:
                    count = 0
                    ichunk += 1
                    chunks.append([])
                gi = base + ichunk * chunk_base + count
                chunks[-1].append(str(gi))
                txn.put(('%09d' % gi).encode(), bytes(frame_bytes), db=db)
                if (count + ichunk * chunk_size) % 1000 == 0:
                    print(f'{prefix} {suffix} frame {count + ichunk*chunk_size}')
                count += 1

        if prev_ichunk is None:
            prev_ichunk = ichunk
            prev_count  = count
        else:
            assert ichunk == prev_ichunk and count == prev_count, (
                f'Suffix mismatch: {ichunk} vs {prev_ichunk}, {count} vs {prev_count}'
            )

    print(f'Processed {len(chunks)} chunks for {prefix}')
    prefix_list.append(prefix)
    idx.append('|'.join(','.join(c) for c in chunks))

    with env.begin(write=True) as txn:
        txn.put(b'N',           str(N + 1).encode())
        txn.put(b'prefix_list', ','.join(prefix_list).encode())
        txn.put(b'idx',         '$'.join(idx).encode())

    env.close()


# ---------------------------------------------------------------------------
# shuffle
# ---------------------------------------------------------------------------

def shuffle(db_path: str, valid_ratio=0.1, test_ratio=0.3):
    """Create train / valid / test index splits and store them in LMDB.

    Movies are split first (test_ratio fraction held out), then
    frames within train movies are further split into train / valid.
    """
    env = lmdb.open(db_path, map_size=1 << 40, max_dbs=5)
    with env.begin() as txn:
        raw_idx  = txn.get(b'idx').decode()
        raw_plist = txn.get(b'prefix_list').decode()
        prefix_list = [p for p in raw_plist.split(',') if p]

    # Parse idx: per-movie list of chunks, each chunk a list of str indices
    idx = [
        [[k for k in chunk.split(',') if k]
         for chunk in movie.split('|') if chunk]
        for movie in raw_idx.split('$') if movie
    ]
    # Drop first and last few chunks (may contain credits / logos)
    idx = [chunks[1:-3] if len(chunks) > 4 else chunks for chunks in idx]

    # Shuffle movie order
    perm = list(range(len(idx)))
    random.shuffle(perm)
    idx         = [idx[i]         for i in perm]
    prefix_list = [prefix_list[i] for i in perm]

    sep = int((1 - test_ratio) * len(idx))
    print('Train movies:', prefix_list[:sep])
    print('Test  movies:', prefix_list[sep:])

    # Test set: all frames from held-out movies
    test_chunks = sum(idx[sep:], [])
    test_idx    = sum(test_chunks, [])
    random.shuffle(test_idx)

    # Train + valid: frames from training movies
    train_chunks = sum(idx[:sep], [])
    random.shuffle(train_chunks)
    vsep = int((1 - valid_ratio) * len(train_chunks))

    valid_idx = sum(train_chunks[vsep:], [])
    random.shuffle(valid_idx)

    train_idx = sum(train_chunks[:vsep], [])
    random.shuffle(train_idx)

    shuffled_test_idx = test_idx[:]
    random.shuffle(shuffled_test_idx)

    print(f'Sizes – train: {len(train_idx)}, valid: {len(valid_idx)}, test: {len(test_idx)}')

    with env.begin(write=True) as txn:
        txn.put(b'train_idx',         ','.join(train_idx).encode())
        txn.put(b'valid_idx',         ','.join(valid_idx).encode())
        txn.put(b'test_idx',          ','.join(test_idx).encode())
        txn.put(b'shuffled_test_idx', ','.join(shuffled_test_idx).encode())

    env.close()
    print('Shuffle complete.')


# ---------------------------------------------------------------------------
# show_db (sanity check)
# ---------------------------------------------------------------------------

def show_db(db_path: str, suffix_list=('l', 'r')):
    """Print summary information about the LMDB database."""
    env = lmdb.open(db_path, map_size=1 << 40, max_dbs=5, readonly=True)
    with env.begin() as txn:
        N          = int(txn.get(b'N'))
        chunk_size = int(txn.get(b'chunk_size'))
        chunk_base = int(txn.get(b'chunk_base'))
        movie_base = int(txn.get(b'movie_base'))
        prefix_list = txn.get(b'prefix_list').decode().split(',')

    print(f'Movies: {N}  chunk_size: {chunk_size}  '
          f'chunk_base: {chunk_base}  movie_base: {movie_base}')
    print('Prefixes:', prefix_list)

    for split_key in (b'train_idx', b'valid_idx', b'test_idx'):
        with env.begin() as txn:
            raw = txn.get(split_key, b'')
        n = len([v for v in raw.split(b',') if v])
        print(f'{split_key.decode()}: {n} samples')

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_cmd(args):
    """Add all .npy movie prefixes found in a folder to the LMDB."""
    folder = args.folder
    db     = args.db
    suffixes = tuple(args.suffixes.split(','))

    # Collect unique prefixes: files named <prefix>_l.npy
    prefixes = set()
    for fname in os.listdir(folder):
        if fname.endswith('_l.npy'):
            prefixes.add(os.path.join(folder, fname[:-6]))  # strip '_l.npy'
    print(f'Found {len(prefixes)} movie prefix(es): {sorted(prefixes)}')
    for prefix in sorted(prefixes):
        add_to_db(db, prefix, suffix_list=suffixes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Manage Deep3D training LMDB database'
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    # add
    p_add = sub.add_parser('add', help='Add movies from a folder to LMDB')
    p_add.add_argument('folder', help='Folder containing <prefix>_l.npy files')
    p_add.add_argument('db',     help='LMDB database path')
    p_add.add_argument('--suffixes', default='l,r',
                       help='Comma-separated sub-db names (default: l,r)')

    # shuffle
    p_shuf = sub.add_parser('shuffle', help='Split LMDB into train/valid/test')
    p_shuf.add_argument('db')
    p_shuf.add_argument('--valid-ratio', type=float, default=0.1)
    p_shuf.add_argument('--test-ratio',  type=float, default=0.3)

    # show
    p_show = sub.add_parser('show', help='Print LMDB summary')
    p_show.add_argument('db')

    args = parser.parse_args()

    if args.cmd == 'add':
        _add_cmd(args)
    elif args.cmd == 'shuffle':
        shuffle(args.db, args.valid_ratio, args.test_ratio)
    elif args.cmd == 'show':
        show_db(args.db)
