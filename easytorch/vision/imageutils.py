import copy as _copy
import math as _math
import os as _os

import cv2 as _cv2
import numpy as _np
from easytorch.utils.logger import *
from easytorch.data.multiproc import multiRun
from PIL import Image as _IMG
import json as _json
import traceback as _tb
from pathlib import Path as _Path

"""
##################################################################################################
Very useful image related utilities
##################################################################################################
"""


def clahe(array, clip_limit=2.0, tile_grid_sz=(8, 8)):
    clahe = _cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_sz)

    _arr = array.copy()
    if len(array.shape) == 2:
        _arr = clahe.apply()

    elif len(array.shape) > 2:
        for i in range(array.shape[2]):
            _arr[:, :, i] = clahe.apply(_arr[:, :, i].copy())
    return _arr


def rescale2d(arr):
    m = _np.max(arr)
    n = _np.min(arr)
    return (arr - n) / (m - n)


def rescale3d(arrays):
    return list(rescale2d(arr) for arr in arrays)


def get_signed_diff_int8(image_arr1=None, image_arr2=None):
    signed_diff = _np.array(image_arr1 - image_arr2, dtype=_np.int8)
    fx = _np.array(signed_diff - _np.min(signed_diff), _np.uint8)
    fx = rescale2d(fx)
    return _np.array(fx * 255, _np.uint8)


def whiten_image2d(img_arr2d=None):
    img_arr2d = img_arr2d.copy()
    img_arr2d = (img_arr2d - img_arr2d.mean()) / img_arr2d.std()
    return _np.array(rescale2d(img_arr2d) * 255, dtype=_np.uint8)


def get_chunk_indexes(img_shape=(0, 0), chunk_shape=(0, 0), offset_row_col=None):
    """
    Returns a generator for four corners of each patch within image as specified.
    :param img_shape: Shape of the original image
    :param chunk_shape: Shape of desired patch
    :param offset_row_col: Offset for each patch on both x, y directions
    :return:
    """
    img_rows, img_cols = img_shape[:2]
    chunk_row, chunk_col = chunk_shape
    offset_row, offset_col = offset_row_col

    row_end = False
    for i in range(0, img_rows, offset_row):
        if row_end:
            continue
        row_from, row_to = i, i + chunk_row
        if row_to > img_rows:
            row_to = img_rows
            row_from = img_rows - chunk_row
            row_end = True

        col_end = False
        for j in range(0, img_cols, offset_col):
            if col_end:
                continue
            col_from, col_to = j, j + chunk_col
            if col_to > img_cols:
                col_to = img_cols
                col_from = img_cols - chunk_col
                col_end = True
            yield [int(row_from), int(row_to), int(col_from), int(col_to)]


def get_chunk_indices_by_index(img_shape=(0, 0), chunk_shape=(0, 0), indices=None):
    x, y = chunk_shape
    ix = []
    for (c1, c2) in indices:
        w, h = img_shape[:2]
        p, q, r, s = c1 - x // 2, c1 + x // 2, c2 - y // 2, c2 + y // 2
        if p < 0:
            p, q = 0, x
        if q > w:
            p, q = w - x, w
        if r < 0:
            r, s = 0, y
        if s > h:
            r, s = h - y, h
        ix.append([int(p), int(q), int(r), int(s)])
    return ix


def merge_patches(patches=None, image_size=(0, 0), patch_size=(0, 0), offset_row_col=None):
    """
    Merge different pieces of image to form a full image. Overlapped regions are averaged.
    :param patches: List of all patches to merge in order (left to right). (N * W * H * C)
    :param image_size: Full image size
    :param patch_size: A patch size(Patches must be uniform in size to be able to merge)
    :param offset_row_col: Offset used to chunk the patches.
    :return:
    """
    padded_sum = _np.zeros(image_size)
    non_zero_count = _np.zeros_like(padded_sum)
    for i, chunk_ix in enumerate(get_chunk_indexes(image_size, patch_size, offset_row_col)):
        row_from, row_to, col_from, col_to = chunk_ix

        patch = _np.array(patches[i]).squeeze()

        _pad = [(row_from, image_size[0] - row_to), (col_from, image_size[1] - col_to)]
        if len(image_size) == 3:
            _pad.append((0, 0))

        padded = _np.pad(patch, _pad, 'constant')
        padded_sum = padded + padded_sum
        non_zero_count = non_zero_count + _np.array(padded > 0).astype(int)
    non_zero_count[non_zero_count == 0] = 1
    return _np.array(padded_sum / non_zero_count, dtype=_np.uint8)


def expand_and_mirror_patch(full_img_shape=None, orig_patch_indices=None, expand_by=None):
    """
    Given a patch within an image, this function select a speciified region around it if present, else mirros it.
    It is useful in neuralnetworks like u-net which look for wide range of area than the actual input image.
    :param full_img_shape: Full image shape
    :param orig_patch_indices: Four cornets of the actual patch
    :param expand_by: Expand by (x, y ) in each dimension
    :return:
    """

    i, j = int(expand_by[0] / 2), int(expand_by[1] / 2)
    p, q, r, s = orig_patch_indices
    a, b, c, d = p - i, q + i, r - j, s + j
    pad_a, pad_b, pad_c, pad_d = [0] * 4
    if a < 0:
        pad_a = i - p
        a = 0
    if b > full_img_shape[0]:
        pad_b = b - full_img_shape[0]
        b = full_img_shape[0]
    if c < 0:
        pad_c = j - r
        c = 0
    if d > full_img_shape[1]:
        pad_d = d - full_img_shape[1]
        d = full_img_shape[1]
    return a, b, c, d, [(pad_a, pad_b), (pad_c, pad_d)]


def largest_cc(binary_arr=None):
    from skimage.measure import label
    labels = label(binary_arr)
    if labels.max() != 0:  # assume at least 1 CC
        largest = labels == _np.argmax(_np.bincount(labels.flat)[1:]) + 1
        return largest


def map_img_to_img2d(map_to, img):
    arr = map_to.copy()

    rgb = arr.copy()
    if len(arr.shape) == 2:
        rgb = _np.zeros((arr.shape[0], arr.shape[1], 3), dtype=_np.uint8)
        rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = arr, arr, arr

    rgb[:, :, 0][img == 255] = 255
    rgb[:, :, 1][img == 255] = 0
    rgb[:, :, 2][img == 255] = 0
    return rgb


def remove_connected_comp(segmented_img, connected_comp_diam_limit=20):
    """
    Remove connected components of a binary image that are less than smaller than specified diameter.
    :param segmented_img: Binary image.
    :param connected_comp_diam_limit: Diameter limit
    :return:
    """

    from scipy.ndimage.measurements import label

    img = segmented_img.copy()
    structure = _np.ones((3, 3), dtype=_np.int)
    labeled, n_components = label(img, structure)
    for i in range(n_components):
        ixy = _np.array(list(zip(*_np.where(labeled == i))))
        x1, y1 = ixy[0]
        x2, y2 = ixy[-1]
        dst = _math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if dst < connected_comp_diam_limit:
            for u, v in ixy:
                img[u, v] = 0
    return img


def get_pix_neigh(i, j, eight=False):
    """
    Get four/ eight neighbors of an image.
    :param i: x position of pixel
    :param j: y position of pixel
    :param eight: Eight neighbors? Else four
    :return:
    """

    n1 = (i - 1, j - 1)
    n2 = (i - 1, j)
    n3 = (i - 1, j + 1)
    n4 = (i, j - 1)
    n5 = (i, j + 1)
    n6 = (i + 1, j - 1)
    n7 = (i + 1, j)
    n8 = (i + 1, j + 1)
    if eight:
        return [n1, n2, n3, n4, n5, n6, n7, n8]
    else:
        return [n2, n5, n7, n4]


def binarize(arr, thr=50, max=255):
    _arr = arr.copy()
    if _arr.max() > thr:
        _arr[_arr < thr] = 0
        _arr[_arr >= thr] = max

    return _arr


def masked_bboxcrop(arr, *apply_to, offset=21, threshold=10):
    """
    Binarize, mask, bbox crop image for largest connected component.
    """
    mask = arr.copy()
    if len(mask.shape) > 2:
        mask = mask[:, :, 1]

    mask[mask < threshold] = 0
    mask[mask >= threshold] = 255

    coords = _cv2.findNonZero(mask)
    x, y, w, h = _cv2.boundingRect(coords)
    a, b, c, d = y, y + h, x, x + w
    a, b, c, d = max(0, a - offset), min(arr.shape[0], b + offset), max(0, c - offset), min(arr.shape[1], d + offset)

    res = [arr[a:b, c:d]]

    for dat in apply_to:
        res.append(dat[a:b, c:d])
    res.append(mask[a:b, c:d])

    return res


def resize(array, size):
    img = _IMG.fromarray(array)
    down = any([a < b for a, b in zip(size, array.shape[:2])])
    array = _IMG.fromarray(array)

    if down:
        array.thumbnail(size, _IMG.ANTIALIAS)
    else:
        array = array.resize((int(size[0]), int(img.size[1] / img.size[0] * size[0])))
    return _np.array(array)


def top_k_larges_cc(arr, k=1, min_comp_size=None, index_only=False):
    from scipy import ndimage
    labeled, nr_objects = ndimage.label(arr)
    comps = [list(zip(*_np.where(labeled == i))) for i in range(1, nr_objects + 1)]

    if min_comp_size:
        comps = sorted([c for c in comps if len(c) > min_comp_size], key=len, reverse=True)[:k]
    else:
        comps = sorted(comps, key=len, reverse=True)[:k]

    if index_only:
        return comps

    res = _np.zeros_like(arr)
    for c in comps:
        c = _np.array(c)
        res[c[:, 0], c[:, 1]] = 255

    return res
