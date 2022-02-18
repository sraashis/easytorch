import numpy as _np
import cv2 as _cv2
import random as _random
from PIL import Image as _IMG


class RandomGaussJitter:
    def __init__(self, gauss_factor=0.2, jitter_factor=0.3):
        self.gauss_factor = gauss_factor
        self.jitter_factor = jitter_factor
        self.gauss_factor = _np.arange(0, self.gauss_factor + 0.01, 0.05)
        self.jitter_factor = _np.arange(0, self.jitter_factor + 0.01, 0.05)

    def __call__(self, img):
        arr = _np.array(img) / 255.0
        noise = _np.abs(_np.random.rand(*arr.shape)) * _random.choice(self.gauss_factor)
        jitt = _np.random.randint(0, 2, arr.shape)
        noise[jitt == 1] *= _random.choice(self.jitter_factor)
        arr = ((arr + noise) * 255).astype(_np.uint8)

        pick = _random.random()
        if pick > 0.67:
            arr = _cv2.GaussianBlur(arr, (3, 3), 0)
            arr = _cv2.medianBlur(arr, 3)
        elif pick > 0.33:
            arr = _cv2.medianBlur(arr, 3)
            arr = _cv2.GaussianBlur(arr, (3, 3), 0)
        else:
            arr = _cv2.GaussianBlur(arr, (3, 3), 0)

        return _IMG.fromarray(arr)
