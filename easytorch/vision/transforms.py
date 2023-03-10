import numpy as _np
import cv2 as _cv2
import random as _random
from PIL import Image as _IMG


class RandomGaussJitter:
    def __init__(self, gauss_factor=0.3, jitter_factor=0.1, p=0.35):
        self.gauss_factor = gauss_factor
        self.jitter_factor = jitter_factor
        self.gauss_factor = _np.arange(0, self.gauss_factor + 0.01, 0.005)
        self.jitter_factor = _np.arange(0, self.jitter_factor + 0.01, 0.005)
        self.p = p

    def __call__(self, img):
        arr = _np.array(img)
        if _random.random() > self.p:
            return _IMG.fromarray(arr)

        arr = arr / 255.0
        noise = _np.abs(_np.random.rand(*arr.shape)) * _random.choice(self.gauss_factor)
        jitt = _np.random.randint(0, 2, arr.shape)
        noise[jitt == 1] *= _random.choice(self.jitter_factor)
        arr = ((arr + noise) * 255).astype(_np.uint8)

        if _random.random() < self.p:
            arr = _cv2.GaussianBlur(arr, (3, 3), 0)

        gray = _cv2.cvtColor(arr, _cv2.COLOR_BGR2GRAY)
        gray = _cv2.merge([gray, gray, gray])
        return _IMG.fromarray(gray)
