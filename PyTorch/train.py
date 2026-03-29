"""
Training script for Deep3D (PyTorch version).

Usage
-----
python train.py [--data DATA_DIR] [--batch-size N] [--epochs N]
                [--lr LR] [--output EXP_PREFIX] [--gpu GPU_ID]
                [--pretrained] [--resume CHECKPOINT]
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import Mov3dDataset, compute_mean
from model import Deep3D, MAELoss


def get_args():
    parser = argparse.ArgumentParser(description='Train Deep3D')
    parser.add_argument('--data',       type=str,   default='data/lmdb',
                        help='Path to LMDB data directory')
    parser.add_argument('--batch-size', type=int,   default=64)
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--lr',         type=float, default=2e-3)
    parser.add_argument('--lr-step',    type=int,   default=20,
                        help='LR decay every this many epochs')
    parser.add_argument('--lr-gamma',   type=float, default=0.1)
    parser.add_argument('--wd',         type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--momentum',   type=float, default=0.9)
    parser.add_argument('--output',     type=str,   default='exp/deep3d',
                        help='Checkpoint prefix')
    parser.add_argument('--gpu',        type=int,   default=0,
                        help='GPU id (-1 for CPU)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Initialise encoder with VGG-16 ImageNet weights')
    parser.add_argument('--resume',     type=str,   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num-workers',type=int,   default=4)
    parser.add_argument('--log-every',  type=int,   default=10,
                        help='Log training loss every N batches')
    parser.add_argument('--val-batches',type=int,   default=100,
                        help='Number of validation batches per epoch')
    return parser.parse_args()


def setup_logging(prefix: str):
    ts = datetime.now().strftime('_%Y_%m_%d-%H_%M.log')
    log_path = prefix + ts
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(),
        ]
    )


def save_checkpoint(model, optimizer, epoch: int, prefix: str):
    os.makedirs(os.path.dirname(prefix) if os.path.dirname(prefix) else '.', exist_ok=True)
    path = f'{prefix}-{epoch:04d}.pth'
    torch.save({
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
    }, path)
    logging.info(f'Checkpoint saved to {path}')


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0)


def validate(model, loader, criterion, device, max_batches: int):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch_idx, (left, left0, right) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            left  = left.to(device)
            left0 = left0.to(device)
            right = right.to(device)
            pred  = model(left, left0)
            loss  = criterion(pred, right)
            total_loss += loss.item()
            n += 1
    return total_loss / n if n > 0 else float('nan')


def train():
    args = get_args()
    setup_logging(args.output)
    logging.info(f'Args: {args}')

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device: {device}')

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------
    data_shape = (384, 160)   # (W, H), matching original config
    scale = (-15, 17)

    # Compute or load channel mean
    mean_path = os.path.join(args.data, 'mean.npy')
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
        logging.info(f'Loaded mean from {mean_path}: {mean}')
    else:
        logging.info('Computing channel mean from training data …')
        mean = compute_mean(args.data, data_shape, n_samples=5000)
        np.save(mean_path, mean)
        logging.info(f'Mean saved to {mean_path}: {mean}')

    train_dataset = Mov3dDataset(args.data, data_shape=data_shape,
                                 scale=scale, split='train', mean=mean)
    valid_dataset = Mov3dDataset(args.data, data_shape=data_shape,
                                 scale=scale, split='valid', mean=mean,
                                 test_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False)

    logging.info(f'Train samples: {len(train_dataset)}, '
                 f'Valid samples: {len(valid_dataset)}')

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    model = Deep3D(scale=scale, upsample=1).to(device)

    if args.pretrained:
        logging.info('Loading VGG-16 pretrained encoder weights …')
        model.load_vgg16_weights(pretrained=True)

    # -----------------------------------------------------------------
    # Optimiser & scheduler
    # -----------------------------------------------------------------
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.lr_step,
                                          gamma=args.lr_gamma)
    criterion = MAELoss()

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer)
        logging.info(f'Resumed from epoch {start_epoch}')

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        epoch_loss   = 0.0
        n_batches    = 0

        for batch_idx, (left, left0, right) in enumerate(train_loader):
            left  = left.to(device)
            left0 = left0.to(device)
            right = right.to(device)

            optimizer.zero_grad()
            pred = model(left, left0)
            loss = criterion(pred, right)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss   += loss.item()
            n_batches    += 1

            if (batch_idx + 1) % args.log_every == 0:
                avg = running_loss / args.log_every
                logging.info(
                    f'Epoch [{epoch+1}/{args.epochs}] '
                    f'Batch [{batch_idx+1}/{len(train_loader)}] '
                    f'Loss: {avg:.4f} '
                    f'LR: {scheduler.get_last_lr()[0]:.6f}'
                )
                running_loss = 0.0

        avg_train = epoch_loss / n_batches if n_batches > 0 else float('nan')
        avg_val   = validate(model, valid_loader, criterion, device,
                             args.val_batches)
        logging.info(
            f'Epoch [{epoch+1}/{args.epochs}] '
            f'Train MAE: {avg_train:.4f}  Val MAE: {avg_val:.4f}'
        )

        scheduler.step()
        save_checkpoint(model, optimizer, epoch + 1, args.output)

    logging.info('Training complete.')


if __name__ == '__main__':
    train()
