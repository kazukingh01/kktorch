import cv2
import numpy as np
from PIL import Image
import torchvision


__all__ = [
    "fit_resize",
    "pil2cv",
    "cv2pil",
    "gray_to_rgb",
]


def fit_resize(img: np.ndarray, dim: str, scale: int):
    """
    Params::
        img: image
        dim: x or y
        scale: width or height
    """
    if dim not in ["x","y"]: raise Exception(f"dim: {dim} is 'x' or 'y'.")
    height = img.shape[0]
    width  = img.shape[1]
    height_after, width_after = None, None
    if   type(scale) == int and scale > 10:
        if   dim == "x":
            width_after  = int(scale)
            height_after = int(height * (scale / width))
        elif dim == "y":
            height_after = int(scale)
            width_after  = int(width * (scale / height))
    else:
        raise Exception(f"scale > 10.")
    img = cv2.resize(img , (width_after, height_after)) # w, h
    return img

def pil2cv(img: Image) -> np.ndarray:
    new_image = np.array(img, dtype=np.uint8)
    if new_image.ndim == 2:  # gray
        pass
    elif new_image.shape[2] == 3:  # RGB
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # RGBA
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(img: np.ndarray):
    new_image = img.copy()
    if new_image.ndim == 2:  # gray
        pass
    elif new_image.shape[2] == 3:  # RGB
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # RGBA
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def gray_to_rgb(img: np.ndarray):
    return np.concatenate([img.reshape(img.shape[0], img.shape[1], 1).copy() for _ in range(3)], axis=2)

