import json
import numpy as np
import cv2
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset

from kktorch.util.com import correct_dirpath
import kktorch.util.image as proc_image


class ImageDataset(Dataset):
    def __init__(
        self, json_data: str, root_dirpath: str=None,
        str_filename: str="name", str_label: str="label",
        transforms = ["pil2cv", ]
    ):
        super().__init__()
        """
        label infomation load
        Format::
            [
                {"name": "test0.png", "label": 1}, 
                {"name": "test0.png", "label": [1,2,3,]}, 
                ...
            ]
        Params::
            transforms: like augmentations.
        """
        self.str_filename = str_filename
        self.str_label    = str_label
        self.json_data   = json.load(open(json_data)) if type(json_data) == str else json_data.copy()
        for x in self.json_data:
            x[self.str_label] = tuple(x[self.str_label]) if type(x[self.str_label]) in [list, tuple] else (x[self.str_label],)
        self.root_dirpath = correct_dirpath(root_dirpath) if root_dirpath is not None else None
        self.transforms   = transforms if isinstance(transforms, list) else []
        self.transforms   = [getattr(proc_image, proc_name) for proc_name in self.transforms]
        self.len          = len(self.json_data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        name   = self.json_data[index][self.str_filename]
        img    = Image.open(name if self.root_dirpath is None else self.root_dirpath + name)
        for proc in self.transforms: img = proc(img)
        labels = self.json_data[index][self.str_label]
        return img, labels
