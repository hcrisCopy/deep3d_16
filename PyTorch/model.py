"""
Deep3D model implemented in PyTorch.

Architecture:
  - A simplified VGG-like encoder (5 conv groups + 2 FC layers) whose conv
    weights can be initialised from a torchvision VGG-16 pretrained checkpoint.
  - Multi-scale prediction heads at each pooling level (pool1–pool5/FC).
  - A custom DepthDot layer that reconstructs the right-eye view from the
    left-eye image weighted by the predicted per-pixel disparity distribution.
  - L1 (MAE) training loss.

The disparity scale is (s0, s1) = (-15, 17), giving 33 channels.  Following
the original CUDA kernel the loop runs over j ∈ [s0, s1) (i.e. 32 values),
using softmax channels 1 … 32 while channel 0 is unused – this behaviour is
preserved here to keep numerical parity with the original model.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DepthDot
# ---------------------------------------------------------------------------

class DepthDotFunction(torch.autograd.Function):
    """Pure-PyTorch implementation of the DepthDot forward/backward pass.

    Forward
    -------
    Given
        softmax : (B, D, H, W)   disparity distribution (already softmax-ed)
        left0   : (B, C, H, W)   left-eye image
        s0, s1  : int             disparity range [s0, s1)
    the output is:
        out[b, c, h, w] = Σ_{j=s0}^{s1-1}  softmax[b, j-s0+1, h, w]
                                             * left0[b, c, h, clamp(w-j, 0, W-1)]

    Boundary condition: clamp (replicate-pad) at left/right edges, matching
    the original CUDA kernel.
    """

    @staticmethod
    def forward(ctx, softmax, left0, s0, s1):
        B, D, H, W = softmax.shape
        device = softmax.device
        output = torch.zeros(B, left0.shape[1], H, W, device=device,
                             dtype=softmax.dtype)
        w_idx = torch.arange(W, device=device, dtype=torch.long)
        for j in range(s0, s1):                  # s0 … s1-1 inclusive
            d_idx = j - s0 + 1                   # channel index into softmax
            src_idx = (w_idx - j).clamp(0, W - 1)
            shifted = left0[:, :, :, src_idx]    # (B, C, H, W)
            weight = softmax[:, d_idx, :, :].unsqueeze(1)  # (B,1,H,W)
            output = output + weight * shifted
        ctx.save_for_backward(softmax, left0)
        ctx.s0 = s0
        ctx.s1 = s1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        softmax, left0 = ctx.saved_tensors
        s0, s1 = ctx.s0, ctx.s1
        B, D, H, W = softmax.shape
        device = softmax.device
        grad_softmax = torch.zeros_like(softmax)
        w_idx = torch.arange(W, device=device, dtype=torch.long)
        for j in range(s0, s1):
            d_idx = j - s0 + 1
            src_idx = (w_idx - j).clamp(0, W - 1)
            shifted = left0[:, :, :, src_idx]    # (B, C, H, W)
            # grad w.r.t. softmax channel d_idx: sum over C of grad * shifted
            grad_softmax[:, d_idx, :, :] = (grad_output * shifted).sum(dim=1)
        # grad w.r.t. left0 is not needed (it is treated as a constant input
        # in the original code; the label gradient is kNullOp in MXNet)
        return grad_softmax, None, None, None


def depth_dot(softmax: torch.Tensor, left0: torch.Tensor,
              s0: int, s1: int) -> torch.Tensor:
    """Thin wrapper around :class:`DepthDotFunction`."""
    return DepthDotFunction.apply(softmax, left0, s0, s1)


# ---------------------------------------------------------------------------
# Bilinear deconvolution initialiser
# ---------------------------------------------------------------------------

def _bilinear_kernel(kernel_size: int) -> torch.Tensor:
    """Return a bilinear upsampling kernel of shape (kernel_size, kernel_size)."""
    f = math.ceil(kernel_size / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] / f - c)) * (1 - abs(og[1] / f - c))
    return torch.from_numpy(filt.astype(np.float32))


def init_bilinear_deconv(m: nn.ConvTranspose2d) -> None:
    """In-place bilinear initialisation for a ConvTranspose2d layer."""
    n_out, n_in, kH, kW = m.weight.shape
    assert n_out == n_in, "Bilinear init requires equal in/out channels."
    kernel = _bilinear_kernel(kH)
    with torch.no_grad():
        m.weight.zero_()
        for i in range(n_out):
            m.weight[i, i] = kernel


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_c, out_c, kernel=3, pad=1, bias=True):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel, padding=pad, bias=bias),
            nn.ReLU(inplace=True),
        )


class BnConv(nn.Sequential):
    """BatchNorm → Conv2d (no activation) used on pool feature maps."""
    def __init__(self, in_c, out_c, kernel=3, pad=1):
        super().__init__(
            nn.BatchNorm2d(in_c),
            nn.Conv2d(in_c, out_c, kernel, padding=pad, bias=True),
        )


# ---------------------------------------------------------------------------
# Deep3D model
# ---------------------------------------------------------------------------

class Deep3D(nn.Module):
    """End-to-end 2D→3D conversion network.

    Parameters
    ----------
    scale : tuple(int, int)
        Disparity range (s0, s1).  ``num_disp = s1 - s0 + 1`` channels are
        predicted; the loop runs over [s0, s1) preserving the original
        behaviour.
    upsample : int
        Upsampling factor applied to *left0* during inference.  For training
        keep ``upsample=1``.
    """

    def __init__(self, scale=(-15, 17), upsample: int = 1):
        super().__init__()
        self.s0 = scale[0]
        self.s1 = scale[1]
        self.num_disp = scale[1] - scale[0] + 1   # 33
        self.upsample = upsample

        # ---- Encoder (simplified VGG-16 style) ----------------------------
        # Group 1: 3 → 64
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Group 2: 64 → 128
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Group 3: 128 → 256
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Group 4: 256 → 512
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Group 5: 512 → 512
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, 2)

        # FC layers (spatial size after 5 pools on 160×384 input = 5×12)
        self.fc6 = nn.Linear(512 * 5 * 12, 512)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(512, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(512, self.num_disp * 5 * 12)

        # ---- Prediction heads (one per scale) ------------------------------
        D = self.num_disp

        # pred1: pool1 (64 ch) → BN → conv → ReLU → deconv stride-1
        self.bn_pool1 = nn.BatchNorm2d(64)
        self.pred1_conv = nn.Conv2d(64, D, 3, padding=1)
        self.deconv_pred1 = nn.ConvTranspose2d(D, D, 1, stride=1, padding=0)

        # pred2: pool2 (128 ch) → BN → conv → ReLU → deconv stride-2
        self.bn_pool2 = nn.BatchNorm2d(128)
        self.pred2_conv = nn.Conv2d(128, D, 3, padding=1)
        self.deconv_pred2 = nn.ConvTranspose2d(D, D, 4, stride=2, padding=1)

        # pred3: pool3 (256 ch) → BN → conv → ReLU → deconv stride-4
        self.bn_pool3 = nn.BatchNorm2d(256)
        self.pred3_conv = nn.Conv2d(256, D, 3, padding=1)
        self.deconv_pred3 = nn.ConvTranspose2d(D, D, 8, stride=4, padding=2)

        # pred4: pool4 (512 ch) → BN → conv → ReLU → deconv stride-8
        self.bn_pool4 = nn.BatchNorm2d(512)
        self.pred4_conv = nn.Conv2d(512, D, 3, padding=1)
        self.deconv_pred4 = nn.ConvTranspose2d(D, D, 16, stride=8, padding=4)

        # pred5: FC output reshaped (5×12) → ReLU → deconv stride-16
        self.deconv_pred5 = nn.ConvTranspose2d(D, D, 32, stride=16, padding=8)

        # ---- Final upsampling branch (×2) ----------------------------------
        self.deconv_predup = nn.ConvTranspose2d(D, D, 4, stride=2, padding=1)
        self.pred_final_conv = nn.Conv2d(D, D, 3, padding=1)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        # Default init for all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Bilinear init for upsampling deconv layers
        for m in [self.deconv_pred2, self.deconv_pred3,
                  self.deconv_pred4, self.deconv_pred5, self.deconv_predup]:
            init_bilinear_deconv(m)

    def load_vgg16_weights(self, pretrained: bool = True):
        """Copy matching conv weights from a torchvision VGG-16 model.

        Parameters
        ----------
        pretrained : bool
            If True, load ImageNet-pretrained weights; otherwise use random.
        """
        import torchvision.models as tvm
        weights = tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = tvm.vgg16(weights=weights)

        # Map: (this model layer, vgg16 feature index)
        mapping = [
            (self.conv1_1, 0),
            (self.conv2_1, 5),
            (self.conv3_1, 10),
            (self.conv3_2, 12),
            (self.conv4_1, 17),
            (self.conv4_2, 19),
            (self.conv5_1, 24),
            (self.conv5_2, 26),
        ]
        with torch.no_grad():
            for our_layer, vgg_idx in mapping:
                vgg_layer = vgg.features[vgg_idx]
                our_layer.weight.copy_(vgg_layer.weight)
                our_layer.bias.copy_(vgg_layer.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor):
        """Run encoder and return intermediate feature maps."""
        x = self.relu1_1(self.conv1_1(x))
        p1 = self.pool1(x)                           # H/2

        x = self.relu2_1(self.conv2_1(p1))
        p2 = self.pool2(x)                           # H/4

        x = self.relu3_1(self.conv3_1(p2))
        x = self.relu3_2(self.conv3_2(x))
        p3 = self.pool3(x)                           # H/8

        x = self.relu4_1(self.conv4_1(p3))
        x = self.relu4_2(self.conv4_2(x))
        p4 = self.pool4(x)                           # H/16

        x = self.relu5_1(self.conv5_1(p4))
        x = self.relu5_2(self.conv5_2(x))
        p5 = self.pool5(x)                           # H/32

        return p1, p2, p3, p4, p5

    def predict_disp(self, p1, p2, p3, p4, p5) -> torch.Tensor:
        """Predict multi-scale disparity map (before softmax)."""
        B = p1.shape[0]

        # FC branch (pred5)
        fc = p5.view(B, -1)
        fc = self.drop6(self.relu6(self.fc6(fc)))
        fc = self.drop7(self.relu7(self.fc7(fc)))
        fc = self.fc8(fc)
        pred5 = fc.view(B, self.num_disp, 5, 12)

        # Scale-1 prediction from pool1 (stride-1 deconv)
        x1 = F.relu(self.pred1_conv(self.bn_pool1(p1)))
        x1 = self.deconv_pred1(x1)

        # Scale-2 prediction from pool2
        x2 = F.relu(self.pred2_conv(self.bn_pool2(p2)))
        x2 = self.deconv_pred2(x2)

        # Scale-3 prediction from pool3
        x3 = F.relu(self.pred3_conv(self.bn_pool3(p3)))
        x3 = self.deconv_pred3(x3)

        # Scale-4 prediction from pool4
        x4 = F.relu(self.pred4_conv(self.bn_pool4(p4)))
        x4 = self.deconv_pred4(x4)

        # Scale-5 prediction from FC output
        x5 = F.relu(pred5)
        x5 = self.deconv_pred5(x5)

        # Crop / align all maps to x1's spatial size
        h, w = x1.shape[2], x1.shape[3]
        def _crop(t):
            return t[:, :, :h, :w]

        feat = _crop(x1) + _crop(x2) + _crop(x3) + _crop(x4) + _crop(x5)
        feat = F.relu(feat)

        # Final ×2 upsample
        up = self.deconv_predup(feat)
        up = F.relu(up)
        up = up[:, :, :h * 2, :w * 2]
        up = self.pred_final_conv(up)

        return up   # (B, D, H_in, W_in)

    def forward(self, left: torch.Tensor,
                left0: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        left  : (B, 3, H, W)          left-eye image (mean-subtracted)
        left0 : (B, 3, H*up, W*up)    left-eye image (original scale, used
                                       for DepthDot reconstruction)

        Returns
        -------
        pred  : (B, 3, H*up, W*up)    reconstructed right-eye view
        """
        p1, p2, p3, p4, p5 = self.encode(left)
        disp_logits = self.predict_disp(p1, p2, p3, p4, p5)

        softmax = F.softmax(disp_logits, dim=1)   # (B, D, H, W)

        if self.upsample > 1:
            # Bilinearly upsample softmax to match left0's spatial size
            softmax = F.interpolate(softmax, scale_factor=self.upsample,
                                    mode='bilinear', align_corners=False)

        pred = depth_dot(softmax, left0, self.s0, self.s1)
        return pred

    def get_softmax(self, left: torch.Tensor) -> torch.Tensor:
        """Return the per-pixel disparity distribution (for visualisation)."""
        p1, p2, p3, p4, p5 = self.encode(left)
        disp_logits = self.predict_disp(p1, p2, p3, p4, p5)
        return F.softmax(disp_logits, dim=1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class MAELoss(nn.Module):
    """Mean Absolute Error (L1) loss, matching mx.metric.MAE."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)
