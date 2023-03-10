import numpy as _np
import cv2 as _cv2
import random as _random
from PIL import Image as _IMG


class RandomGaussJitter:
    def __init__(self, gauss_factor=0.1, jitter_factor=0.2, p=0.35):
        self.gauss_factor = gauss_factor
        self.jitter_factor = jitter_factor
        self.gauss_factor = _np.arange(0, self.gauss_factor + 0.01, 0.05)
        self.jitter_factor = _np.arange(0, self.jitter_factor + 0.01, 0.05)
        self.p = p

    def __call__(self, img):
        array = _np.array(img)
        if len(array.shape) == 2:
            array = array[..., None]

        for i in range(array.shape[2]):
            arr = array[:, :, i].copy()
            if _random.random() < self.p:
                arr = arr / 255.0
                noise = _np.abs(_np.random.rand(*arr.shape)) * _random.choice(self.gauss_factor)
                jitt = _np.random.randint(0, 2, arr.shape)
                noise[jitt == 1] *= _random.choice(self.jitter_factor)
                arr = ((arr + noise) * 255).astype(_np.uint8)

            if _random.random() < self.p:
                arr = _cv2.GaussianBlur(arr, (3, 3), 0)
                arr = _cv2.medianBlur(arr, 3)

            if _random.random() < self.p:
                arr = _cv2.medianBlur(arr, 3)
                arr = _cv2.GaussianBlur(arr, (3, 3), 0)

            if _random.random() < self.p:
                arr = _cv2.GaussianBlur(arr, (3, 3), 0)
            array[:, :, i] = arr

        return _IMG.fromarray(array.squeeze())
