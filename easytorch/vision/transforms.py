import random as _random

import cv2 as _cv2
import numpy as _np
from PIL import Image as _IMG
from skimage.util import random_noise as _noise


class RandomNoise:
    def __init__(self, p=0.5, random_crop=False, **kw):
        self.p = p
        self.noises = ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']
        self.amount = _np.arange(0.01, 0.08, 0.001)
        self.vars = _np.arange(0.01, 0.03, 0.001)
        self.means = _np.arange(0.0, 0.03, 0.001)
        self.random_crop = random_crop

    def _random_crop_ix(self, arr):
        h, w, _ = arr.shape
        crop_height = _random.choice(list(range(h // 10, h - 1, 5)))
        crop_width = _random.choice(list(range(w // 10, w - 1, 5)))

        max_x = arr.shape[1] - crop_width
        max_y = arr.shape[0] - crop_height

        x = _np.random.randint(0, max_x)
        y = _np.random.randint(0, max_y)
        return [y, y + crop_height, x, x + crop_width]

    def _crop_mask(self, arr, pqrs):
        p, q, r, s = pqrs
        mask = _np.zeros_like(arr)
        mask[p:q, r:s] = 255
        return

    def __call__(self, img):
        arr = _np.array(img)
        if _random.random() > self.p:
            return _IMG.fromarray(arr)

        args = {}
        noise = _random.choice(self.noises)
        if noise in ['salt', 'pepper', 's&p']:
            args['amount'] = _random.choice(self.amount)

        if noise in ['gaussian', 'speckle']:
            args['mean'] = _random.choice(self.means)
            args['var'] = _random.choice(self.vars)

            if noise == 'speckle':
                """speckle is weak"""
                args['mean'] *= 2
                args['var'] *= 2

        _arr = arr.copy()
        noisy = (_noise(_arr, noise, **args) * 255).astype(_np.uint8)
        if _random.random() <= self.p and self.random_crop:
            p, q, r, s = self._random_crop_ix(arr)
            _arr[p:q, r:s] = noisy[p:q, r:s]
        else:
            _arr = noisy

        gray = _cv2.cvtColor(_arr, _cv2.COLOR_BGR2GRAY)
        gray = _cv2.merge([gray, gray, gray])
        gray = _cv2.GaussianBlur(gray, (3, 3), 0)
        return _IMG.fromarray(gray)
