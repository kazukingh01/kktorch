import pandas as pd
import cv2
from typing import List, Union
from PIL import Image
import torch
from torch.utils.data import Dataset

from kkannotation.streamer import Streamer
from kktorch.util.com import check_type_list
import kktorch.util.image as tfms


__all__ = [
    "ImageDataset",
    "DataframeDataset",
    "VideoDataset",
]


class ImageDataset(Dataset):
    def __init__(
        self, image_paths: List[str], labels: List[object],
        transforms: Union[tfms.Compose, List[tfms.Compose]]=tfms.Compose([tfms.ToTensor(),])
    ):
        super().__init__()
        """
        Params::
            image_paths:
                list of image file paths.
                ["./data/imageA.jpg", "./data/imageB.jpg", ...]
            labels:
                labels.
                [1, 0, ...] or [[1,2,3], [2,3,4], ...] or [objectA, objectB, ...]
            transforms:
                torchvision.transforms.Compose or List
                If Compose:
                    a image is transformed by Compose and create 1 image.
                If List:
                    a image is transformed by Compose of List and create N image.
        """
        assert check_type_list(image_paths, str)
        assert isinstance(labels, list) and len(image_paths) == len(labels)
        assert isinstance(transforms, tfms.Compose) or check_type_list(transforms, tfms.Compose)
        self.image_paths  = image_paths
        self.labels       = labels
        self.len          = len(self.image_paths)
        self.transforms   = [transforms] if isinstance(transforms, tfms.Compose) else transforms
        self.concat       = (lambda x: x) if len(self.transforms) == 1 else self.concat_same_resolution
        self.is_check     = True
        self.dict_indexes = {}
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        path = self.image_paths[index]
        img  = Image.open(path)
        imgs = []
        for procs in self.transforms: imgs.append(procs(img))
        imgs   = self.concat(imgs)
        labels = self.labels[index]
        return (*imgs, labels)
    def concat_same_resolution(self, input: List[torch.Tensor]):
        """
        Combine images of the same resolution when there are multiple augmentations and different resolutions are output.
        """
        if self.is_check:
            assert check_type_list(input, torch.Tensor)
            assert sum([(len(img.shape) == 3) for img in input]) == len(input)
            for i, img in enumerate(input):
                shape = tuple(img.shape)
                if self.dict_indexes.get(shape) is None:
                    self.dict_indexes[shape] = [i]
                else:
                    self.dict_indexes[shape].append(i)
            self.is_check = False
        output = []
        for _, y in self.dict_indexes.items():
            output.append(torch.stack([input[i] for i in y]))
        return output
    def get_image(self, index: int, img_type: str="cv2"):
        assert isinstance(img_type, str) and img_type in ["cv2", "pil"]
        path = self.image_paths[index]
        img  = None
        if   img_type == "cv2":
            img = cv2.imread(path)
        elif img_type == "pil":
            img = Image.open(path)
        return img


class DataframeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, columns: Union[List[str], List[int]], **kwargs):
        assert isinstance(df, pd.DataFrame)
        assert check_type_list(columns, [int, str])
        self.columns = columns
        self.ndf     = df.iloc[:, self.columns].values.copy() if check_type_list(columns, [int]) else df.loc[:, self.columns].values.copy()
        super().__init__(**kwargs)
    def __getitem__(self, index: int):
        return tuple(self.ndf[index])
    def __len__(self):
        return self.ndf.shape[0]


class VideoImageDataset(Dataset):
    def __init__(
        self, video_file_path: str, labels: List[object], 
        reverse: bool=False, start_frame_id: int=0, max_frames: int=None,
        transforms: Union[tfms.Compose, List[tfms.Compose]]=tfms.Compose([tfms.ToTensor(),]),
        **kwargs    
    ):
        """
        Params::
            video_file_path:
                video file paths.
            labels:
                labels.
                [1, 0, ...] or [[1,2,3], [2,3,4], ...] or [objectA, objectB, ...]
        """
        assert isinstance(video_file_path, str)
        assert isinstance(transforms, tfms.Compose)
        self.streamer = Streamer(video_file_path, reverse=reverse, start_frame_id=start_frame_id, max_frames=max_frames)
        assert isinstance(labels, list) and len(self.streamer) == len(labels)
        self.labels     = labels
        self.transforms = transforms
        super().__init__(**kwargs)
    def __len__(self):
        return len(self.streamer)
    def __getitem__(self, index: int):
        img    = self.streamer[index]
        img    = Image.fromarray(img)
        img    = self.transforms(img)
        labels = self.labels[index]
        return img, labels