import json
import pandas as pd
import numpy as np
import cv2
from typing import List, Union, Callable
from PIL import Image
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


class TextDataset(Dataset):
    def __init__(self, tokenizer: Callable=None, preproc: List[Callable]=None, aftproc: List[Callable]=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.preproc   = preproc if isinstance(preproc, list) else ([] if preproc is None else [preproc, ])
        self.aftproc   = aftproc if isinstance(aftproc, list) else ([] if aftproc is None else [aftproc, ])
    def __getitem__(self, index: int):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    def apply(self, text: str):
        output = text
        for proc in self.preproc: output = proc(output)
        output = self.tokenizer(output)
        for proc in self.aftproc: output = proc(output)
        return output 
    def transform(self, input: object):
        output = None
        if isinstance(input, list) or isinstance(input, tuple):
            output = [self.apply(x) if isinstance(x, str) else x for x in input]
        elif isinstance(input, str):
            output = self.apply(input)
        else:
            output = input
        return output


class DataframeDataset(TextDataset):
    def __init__(self, df: pd.DataFrame, columns: Union[List[str], List[int]], **kwargs):
        assert isinstance(df, pd.DataFrame)
        assert check_type_list(columns, [int, str])
        self.columns = columns
        self.ndf     = df.iloc[:, self.columns].values.copy() if check_type_list(columns, [int]) else df.loc[:, self.columns].values.copy()
        super().__init__(**kwargs)
    def __getitem__(self, index: int):
        return self.transform(tuple(self.ndf[index]))
    def __len__(self):
        return self.ndf.shape[0]


