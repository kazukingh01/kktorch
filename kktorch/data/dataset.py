import json
import pandas as pd
import numpy as np
import cv2
from typing import List, Union
from PIL import Image
from timm import data
from torch.utils.data import Dataset

from kktorch.util.com import correct_dirpath, check_type_list
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


class DataframeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, columns: Union[List[str], List[int]]):
        assert isinstance(df, pd.DataFrame)
        assert check_type_list(columns, [int, str])
        self.columns = columns
        self.ndf     = df.iloc[:, self.columns].values.copy() if check_type_list(columns, [int]) else df.loc[:, self.columns].values.copy()
    def __getitem__(self, index: int):
        return tuple(self.ndf[index])
    def __len__(self):
        return self.ndf.shape[0]