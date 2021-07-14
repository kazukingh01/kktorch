import random
import numpy as np
from PIL import Image
from torchvision import transforms


__all__ = [
    "RandomRotation90",
    "RandomFliplr",
    "ResizeFixRatio",
]


class RandomRotation90(object):
    """Rotate by one of the given angles."""
    def __init__(self):
        self.values = [0, 1, 2, 3]
    def __call__(self, img: np.ndarray):
        value = random.choice(self.values)
        return np.rot90(img, value) if value > 0 else img

class RandomFliplr(object):
    def __init__(self):
        self.values = [False, True]
    def __call__(self, img: np.ndarray):
        value = random.choice(self.values)
        return np.fliplr(img) if value else img

class ResizeFixRatio(object):
    def __init__(self, size: int, fit_type: str="max"):
        self.size = size
        self.fit_type = fit_type
        if not (isinstance(self.fit_type, str) and self.fit_type in ["max", "min", "v", "h"]):
            raise Exception(f"fit_type must be selected in ['max', 'min', 'v', 'h']")
    def __str__(self):
        return f'{self.__class__.__name__}(size: {self.size}. fit_type: {self.fit_type})'
    def __call__(self, img: Image):
        w, h = img.size
        if   self.fit_type == "max":
            w, h = (self.size, self.size * h / w, ) if w > h else (self.size * w / h, self.size, )
            w, h = int(w), int(h)
        elif self.fit_type == "min":
            w, h = (self.size * w / h, self.size) if w > h else (self.size, self.size * h / w, )
            w, h = int(w), int(h)
        elif self.fit_type == "v":
            w, h = self.size * w / h, self.size
            w, h = int(w), int(h)
        elif self.fit_type == "h":
            w, h = self.size, self.size * h / w
            w, h = int(w), int(h)
        return transforms.Resize((h,w))(img)