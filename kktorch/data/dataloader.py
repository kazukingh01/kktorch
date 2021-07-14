from typing import List, Union
import numpy as np
import torch, torchvision
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.transforms as transforms

from kktorch.util.com import check_type_list


__all__ = [
    "BaseDataLoader",
    "RandomDataLoader",
    "NumpyDataLoader",
    "MNISTDataLoader",
]


class BaseDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, dtype_data=torch.float32, dtype_target=None, **kwargs):
        self.dtype_data   = dtype_data
        self.dtype_target = dtype_target
        if kwargs.get("collate_fn") is None: kwargs["collate_fn"] = self.collate_fn
        if kwargs.get("pin_memory") is None: kwargs["pin_memory"] = True
        super().__init__(dataset, **kwargs)
        self.is_check  = True
        self.to_tensor_data  = lambda x: x
        self.to_tensor_label = lambda x: x
    def collate_fn(self, batch):
        input, label = list(zip(*batch))
        if self.is_check:
            if check_type_list(input, [torch.Tensor]): self.to_tensor_data = torch.stack
            else:                                      self.to_tensor_data = torch.Tensor
            if check_type_list(label, [torch.Tensor]): self.to_tensor_label = torch.stack
            else:                                      self.to_tensor_label = torch.Tensor
            self.is_check = False
        input = self.to_tensor_data( input).to(self.dtype_data)
        label = self.to_tensor_label(label).to(self.dtype_target)
        return input, label
    def sample(self):
        return next(iter(self))


class RandomDataLoader(BaseDataLoader):
    def __init__(
        self, n_data: int, n_features: Union[int, List[int]], n_classes: int, 
        target_type: str="cls", **kwargs
    ):
        assert isinstance(target_type, str) and target_type in ["cls", "reg"]
        n_features   = n_features if isinstance(n_features, list) else [n_features, ]
        data         = torch.rand(n_data, *n_features)
        dtype_target = torch.long
        if target_type == "cls":
            label = torch.randint(n_classes, [n_data,])
        else:
            label = torch.rand(n_data, n_classes)
            dtype_target = torch.float32
        dataset = TensorDataset(data, label)
        super().__init__(dataset, dtype_data=torch.float32, dtype_target=dtype_target, **kwargs)


class NumpyDataLoader(BaseDataLoader):
    def __init__(
        self, data: np.ndarray, target: np.ndarray, 
        dtype_data=torch.float32, dtype_target=torch.long,
        **kwargs
    ):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        dataset = TensorDataset(x, y)
        super().__init__(dataset, dtype_data=dtype_data, dtype_target=dtype_target, **kwargs)


class MNISTDataLoader(BaseDataLoader):
    def __init__(
        self, root: str='./data', train: bool=True, download: bool=True, 
        transform=[transforms.ToTensor(), ], 
        dtype_data=torch.float32, dtype_target=torch.long, **kwargs
    ):
        transform = transform if isinstance(transform, list) else []
        transform = transforms.Compose(transform)
        dataset   = torchvision.datasets.MNIST(root=root, train=train, download=download, transform=transform)
        super().__init__(dataset, dtype_data=dtype_data, dtype_target=dtype_target, **kwargs)
