import copy as _copy
import math as _math
import os as _os

import cv2 as _cv2
import numpy as _np
from PIL import Image as _IMG
from easytorch.utils.logger import *

"""
##################################################################################################
Very useful image related utilities
##################################################################################################
"""


def _same_file(x):
    return x


class Image:
    def __init__(self, dtype=_np.uint8):
        self.dir = None
        self.file = None
        self.array = None
        self.mask = None
        self.ground_truth = None
        self.extras = {}
        self.dtype = dtype

    def load(self, dir, file):
        try:
            self.dir = dir
            self.file = file
            self.array = _np.array(_IMG.open(self.path), dtype=self.dtype)
        except Exception as e:
            error('Fail to load file: ' + self.file + ': ' + str(e))

    def load_mask(self, mask_dir=None, fget_mask=_same_file):
        if fget_mask is None:
            fget_mask = _same_file
        try:
            mask_file = fget_mask(self.file)
            self.mask = _np.array(_IMG.open(_os.path.join(mask_dir, mask_file)), dtype=self.dtype)
        except Exception as e:
            error('Fail to load mask: ' + str(e))

    def load_ground_truth(self, gt_dir=None, fget_ground_truth=_same_file):
        if fget_ground_truth is None:
            fget_ground_truth = _same_file
        try:
            gt_file = fget_ground_truth(self.file)
            self.ground_truth = _np.array(_IMG.open(_os.path.join(gt_dir, gt_file)), dtype=self.dtype)
        except Exception as e:
            error('Fail to load ground truth: ' + str(e))

    def get_array(self, dir='', getter=_same_file, file=None):
        if getter is None:
            getter = _same_file
        if not file:
            file = self.file
        arr = _np.array(_IMG.open(_os.path.join(dir, getter(file))), dtype=self.dtype)
        return arr

    def apply_mask(self):
        if self.mask is not None:
            self.array[self.mask == 0] = 0

    def apply_clahe(self, clip_limit=2.0, tile_shape=(8, 8)):
        enhancer = _cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_shape)
        if len(self.array.shape) == 2:
            self.array = enhancer.apply(self.array)
        elif len(self.array.shape) == 3:
            self.array[:, :, 0] = enhancer.apply(self.array[:, :, 0])
            self.array[:, :, 1] = enhancer.apply(self.array[:, :, 1])
            self.array[:, :, 2] = enhancer.apply(self.array[:, :, 2])
        else:
            error('More than three channels')

    def __copy__(self):
        copy_obj = Image()
        copy_obj.file = _copy.copy(self.file)
        copy_obj.array = _copy.copy(self.array)
        copy_obj.mask = _copy.copy(self.mask)
        copy_obj.ground_truth = _copy.copy(self.ground_truth)
        copy_obj.extras = _copy.deepcopy(self.extras)
        copy_obj.dtype = _copy.deepcopy(self.dtype)
        return copy_obj

    @property
    def path(self):
        return _os.path.join(self.dir, self.file)


def get_rgb_scores(arr_2d=None, truth=None):
    """
    Returns a rgb image of pixelwise separation between ground truth and arr_2d
    (predicted image) with different color codes
    Easy when needed to inspect segmentation result against ground truth.
    :param arr_2d:
    :param truth:
    :return:
    """
    arr_rgb = _np.zeros([arr_2d.shape[0], arr_2d.shape[1], 3], dtype=_np.uint8)
    x = arr_2d.copy()
    y = truth.copy()
    x[x == 255] = 1
    y[y == 255] = 1
    xy = x + (y * 2)
    arr_rgb[xy == 3] = [255, 255, 255]
    arr_rgb[xy == 1] = [0, 255, 0]
    arr_rgb[xy == 2] = [255, 0, 0]
    arr_rgb[xy == 0] = [0, 0, 0]
    return arr_rgb


def get_praf1(arr_2d=None, truth=None):
    """
    Returns precision, recall, f1 and accuracy score between two binary arrays upto five precision.
    :param arr_2d:
    :param truth:
    :return:
    """
    x = arr_2d.copy()
    y = truth.copy()
    x[x == 255] = 1
    y[y == 255] = 1
    xy = x + (y * 2)
    tp = xy[xy == 3].shape[0]
    fp = xy[xy == 1].shape[0]
    tn = xy[xy == 0].shape[0]
    fn = xy[xy == 2].shape[0]
    try:
        p = tp / (tp + fp)
    except ZeroDivisionError:
        p = 0

    try:
        r = tp / (tp + fn)
    except ZeroDivisionError:
        r = 0

    try:
        a = (tp + tn) / (tp + fp + fn + tn)
    except ZeroDivisionError:
        a = 0

    try:
        f1 = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1 = 0

    return {
        'Precision': round(p, 5),
        'Recall': round(r, 5),
        'Accuracy': round(a, 5),
        'F1': round(f1, 5)
    }


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
    img_rows, img_cols = img_shape
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
        w, h = img_shape
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
    :param patches: List of all patches to merge in order (left to right).
    :param image_size: Full image size
    :param patch_size: A patch size(Patches must be uniform in size to be able to merge)
    :param offset_row_col: Offset used to chunk the patches.
    :return:
    """
    padded_sum = _np.zeros([image_size[0], image_size[1]])
    non_zero_count = _np.zeros_like(padded_sum)
    for i, chunk_ix in enumerate(get_chunk_indexes(image_size, patch_size, offset_row_col)):
        row_from, row_to, col_from, col_to = chunk_ix

        patch = _np.array(patches[i, :, :]).squeeze()

        padded = _np.pad(patch, [(row_from, image_size[0] - row_to), (col_from, image_size[1] - col_to)],
                         'constant')
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
