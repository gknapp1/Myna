# Library imports
from pathlib import Path
from typing import List, Tuple
import numpy as np
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import torch
import torch.nn.functional as F
import math


def extract_patches_as_strided(
    img: torch.Tensor,
    patch_size: int = 101,
    stride: int = 21,
    padVal: float = torch.nan,
):

    # Get image
    C, D, H, W = img.shape

    # compute number of patches and padding (only pads on the right to get an integer number of strides)
    n_h = math.ceil((H - patch_size) / stride) + 1
    n_w = math.ceil((W - patch_size) / stride) + 1
    pad_h = max(0, (n_h - 1) * stride + patch_size - H)
    pad_w = max(0, (n_w - 1) * stride + patch_size - W)

    # pad H/W
    img_padded = F.pad(img, (0, pad_w, 0, pad_h), value=padVal)  # (C, D, H_pad, W_pad)
    H_pad, W_pad = img_padded.shape[-2:]

    # strides: (H_stride, W_stride, C_stride, D_stride, within_patch_H, within_patch_W)
    shape = (n_h, n_w, C, D, patch_size, patch_size)
    strides = (
        stride * img_padded.stride(2),  # move along H
        stride * img_padded.stride(3),  # move along W
        img_padded.stride(0),  # channels
        img_padded.stride(1),  # depth
        img_padded.stride(2),  # within patch H
        img_padded.stride(3),  # within patch W
    )

    patches = img_padded.as_strided(shape, strides)
    patches = patches.contiguous().view(n_h * n_w, C, D, patch_size, patch_size)

    return patches, n_h, n_w


def recombine_patches_view(pred_patches: torch.Tensor, n_h: int, n_w: int):

    # Make sure shape is okay
    num_patches, out_dim = pred_patches.shape
    assert num_patches == n_h * n_w

    # Unpatchify it
    pred_img = pred_patches.contiguous().view(n_h, n_w, out_dim)

    return pred_img


def get_patch_coordinates_from_index(
    flat_index: int, n_h: int, n_w: int, patch_size: int, stride: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # Convert the 1D index back to a 2D patch grid index (h_idx, w_idx)
    w_idx = flat_index % n_w
    h_idx = flat_index // n_w

    # Actual integer offset from (min_x,min_y) depends on stride
    x = w_idx * stride
    y = h_idx * stride

    return (x, y), (w_idx, h_idx)
