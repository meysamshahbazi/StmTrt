# adaptaed from video analysis

import cv2
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
# import glob
# import os.path as osp
import torch
# import torch.nn as nn 
# from copy import deepcopy
# import time
# import math
# from collections import OrderedDict
# import torch.nn.functional as F


def get_xy_ctr_np(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in numpy)
    """
    batch, fm_height, fm_width = 1, score_size, score_size

    y_list = np.linspace(0., fm_height - 1.,
                         fm_height).reshape(1, fm_height, 1, 1)
    y_list = y_list.repeat(fm_width, axis=2)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, fm_width, 1)
    x_list = x_list.repeat(fm_height, axis=1)
    xy_list = score_offset + np.concatenate((x_list, y_list), 3) * total_stride
    xy_ctr = np.repeat(xy_list, batch, axis=0).reshape(
        batch, -1,
        2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
    # TODO: consider use float32 type from the beginning of this function
    xy_ctr = torch.from_numpy(xy_ctr.astype(np.float32))
    return xy_ctr


def get_box(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)
    xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])
    xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])
    bboxes_pred = torch.cat([xy0, xy1], 2)

    return bboxes_pred

def xywh2cxywh(rect):
    rect = np.array(rect, dtype=np.float32)
    return np.concatenate([
        rect[..., [0]] + (rect[..., [2]] - 1) / 2, rect[..., [1]] +
        (rect[..., [3]] - 1) / 2, rect[..., [2]], rect[..., [3]]
    ],
                          axis=-1)

def cxywh2xyxy(box):
    box = np.array(box, dtype=np.float32)
    return np.concatenate([
        box[..., [0]] - (box[..., [2]] - 1) / 2, box[..., [1]] -
        (box[..., [3]] - 1) / 2, box[..., [0]] +
        (box[..., [2]] - 1) / 2, box[..., [1]] + (box[..., [3]] - 1) / 2
    ],
                          axis=-1)

def imarray_to_tensor(arr):
    r"""
    Transpose & convert from numpy.array to torch.Tensor
    :param arr: numpy.array, (H, W, C)
    :return: torch.Tensor, (1, C, H, W)
    """
    arr = np.ascontiguousarray(
        arr.transpose(2, 0, 1)[np.newaxis, ...], np.float32)
    return torch.from_numpy(arr)

def tensor_to_numpy(t):
    r"""
    Perform naive detach / cpu / numpy process.
    :param t: torch.Tensor, (N, C, H, W)
    :return: numpy.array, (N, C, H, W)
    """
    arr = t.detach().cpu().numpy()
    return arr

def xyxy2cxywh(bbox):
    bbox = np.array(bbox, dtype=np.float32)
    return np.concatenate([(bbox[..., [0]] + bbox[..., [2]]) / 2,
                           (bbox[..., [1]] + bbox[..., [3]]) / 2,
                           bbox[..., [2]] - bbox[..., [0]] + 1,
                           bbox[..., [3]] - bbox[..., [1]] + 1],
                          axis=-1)


def cxywh2xywh(box):
    box = np.array(box, dtype=np.float32)
    return np.concatenate([
        box[..., [0]] - (box[..., [2]] - 1) / 2, box[..., [1]] -
        (box[..., [3]] - 1) / 2, box[..., [2]], box[..., [3]]
    ],
                          axis=-1)

def get_crop_numpy(im: np.ndarray, pos: np.ndarray, sample_sz: np.ndarray, output_sz: np.ndarray = None,
                   mode: str = 'constant', avg_chans=(0, 0, 0), max_scale_change=None):
    """Sample an image patch.

    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """

    # if mode not in ['replicate', 'inside']:
    #     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

    # copy and convert
    posl = pos.astype(np.int32).copy()

    # Get new sample size if forced inside the image
    # if mode == 'inside' or mode == 'inside_major':
    #     pad_mode = 'replicate'
    #     # im_sz = torch.tensor([im.shape[2], im.shape[3]], device=im.device)
    #     # shrink_factor = (sample_sz.float() / im_sz)
    #     im_sz = np.array([im.shape[0], im.shape[1]])
    #     shrink_factor = (sample_sz.astype(np.float) / im_sz)
    #     if mode == 'inside':
    #         shrink_factor = shrink_factor.max()
    #     elif mode == 'inside_major':
    #         shrink_factor = shrink_factor.min()
    #     shrink_factor.clamp_(min=1, max=max_scale_change)
    #     # sample_sz = (sample_sz.float() / shrink_factor).long()
    #     sample_sz = (sample_sz.astype(np.float) / shrink_factor).astype(np.int32)

    # Compute pre-downsampling factor
    if output_sz is not None:
        # resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
        resize_factor = np.min(sample_sz.astype(np.float) / output_sz.astype(np.float)).item()
        df = int(max(int(resize_factor - 0.1), 1))
    else:
        df = int(1)

    # sz = sample_sz.float() / df  # new size
    sz = sample_sz.astype(np.float) / df

    # Do downsampling
    if df > 1:
        os = posl % df  # offset
        posl = (posl - os) // df  # new position
        im2 = im[os[0].item()::df, os[1].item()::df, :]  # downsample
    else:
        im2 = im

    # compute size to crop
    # szl = torch.max(sz.round(), torch.tensor([2.0], dtype=sz.dtype, device=sz.device)).long()
    szl = np.maximum(np.round(sz), 2.0).astype(np.int32)

    # Extract top and bottom coordinates
    tl = posl - (szl - 1) // 2
    br = posl + szl // 2 + 1

    # Shift the crop to inside
    # if mode == 'inside' or mode == 'inside_major':
    #     # im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
    #     # shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
    #     im2_sz = np.array([im2.shape[0], im2.shape[1]], dtype=np.int32)
    #     shift = np.clip(-tl, 0) - np.clip(br - im2_sz, 0)
    #     tl += shift
    #     br += shift

    #     # outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
    #     # shift = (-tl - outside) * (outside > 0).long()
    #     outside = (np.clip(-tl, 0) - np.clip(br - im2_sz, 0)) // 2
    #     shift = (-tl - outside) * (outside > 0).astype(np.int32)
    #     tl += shift
    #     br += shift

    #     # Get image patch
    #     # im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]

    crop_xyxy = np.array([tl[1], tl[0], br[1], br[0]])
    # warpAffine transform matrix
    M_13 = crop_xyxy[0]
    M_23 = crop_xyxy[1]
    M_11 = (crop_xyxy[2] - M_13) / (output_sz[0] - 1)
    M_22 = (crop_xyxy[3] - M_23) / (output_sz[1] - 1)
    mat2x3 = np.array([
        M_11,
        0,
        M_13,
        0,
        M_22,
        M_23,
    ]).reshape(2, 3)
    im_patch = cv2.warpAffine(im2,
                              mat2x3, (output_sz[0], output_sz[1]),
                              flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=tuple(map(int, avg_chans)))
    # Get image coordinates
    patch_coord = df * np.concatenate([tl, br]).reshape(1, 4)
    scale = output_sz / (np.array([br[1] - tl[1] + 1, br[0] - tl[0] + 1]) * df)
    return im_patch, patch_coord, scale



def get_crop_single(im: np.ndarray, target_pos: np.ndarray, target_scale: float, output_sz: int, avg_chans: tuple):
    pos = target_pos[::-1]
    output_sz = np.array([output_sz, output_sz])
    sample_sz = target_scale * output_sz
    im_patch, _, scale_x = get_crop_numpy(im, pos, sample_sz, output_sz, avg_chans=avg_chans)
    return im_patch, scale_x[0].item()


