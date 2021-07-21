from typing import Callable, List, Union
import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from kktorch.util.com import check_type_list


__all__ = [
    "BaseDataLoader",
    "RandomDataLoader",
    "NumpyDataLoader",
]


class BaseDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, dtype_data: torch.dtype, dtype_target: torch.dtype, **kwargs):
        self.dtype_data   = dtype_data
        self.dtype_target = dtype_target
        if kwargs.get("collate_fn") is None: kwargs["collate_fn"] = self.collate_fn
        if kwargs.get("pin_memory") is None: kwargs["pin_memory"] = True
        super().__init__(dataset, **kwargs)
        self.is_check  = True
        self.to_tensor_data  = lambda x: x
        self.to_tensor_label = lambda x: x
        self.to_dtype_data   = lambda x, y: x.to(y)
        self.to_dtype_label  = lambda x, y: x.to(y)
    def collate_fn(self, batch):
        input, label = list(zip(*batch))
        return self.to_tensor(input, label)
    def to_tensor(self, input, label):
        if self.is_check:
            if isinstance(input, dict):
                self.to_tensor_data = lambda x: {y:torch.Tensor(z) for y, z in x.items()}
                if isinstance(self.dtype_data, dict): self.to_dtype_data = lambda x, y: {a:b.to(y[a]) for a, b in x.items()}
                else:                                 self.to_dtype_data = lambda x, y: {a:b.to(y)    for a, b in x.items()}
            elif   check_type_list(input, [torch.Tensor]):
                self.to_tensor_data = torch.stack
            else:
                self.to_tensor_data = torch.Tensor
            if isinstance(label, dict):
                self.to_tensor_label = lambda x: {y:torch.Tensor(z) for y, z in x.items()}
                if isinstance(self.dtype_target, dict): self.to_dtype_label = lambda x, y: {a:b.to(y[a]) for a, b in x.items()}
                else:                                   self.to_dtype_label = lambda x, y: {a:b.to(y)    for a, b in x.items()}
            elif check_type_list(label, [torch.Tensor]):
                self.to_tensor_label = torch.stack
            else:
                self.to_tensor_label = torch.Tensor
            self.is_check = False
        input = self.to_tensor_data(input)
        label = self.to_tensor_label(label)
        input = self.to_dtype_data( input, self.dtype_data)
        label = self.to_dtype_label(label, self.dtype_target)
        return input, label
    def sample(self):
        return next(iter(self))


class TextDataLoader(BaseDataLoader):
    def __init__(
        self, dataset: Dataset, dtype_data: torch.dtype, dtype_target: torch.dtype, 
        tokenizer: Callable, tokenizer_params_input: dict={}, tokenizer_params_label: dict={}, 
        preprocs_input: List[Callable]=None, aftprocs_input: List[Callable]=None,
        preprocs_label: List[Callable]=None, aftprocs_label: List[Callable]=None,
        **kwargs
    ):
        """
        Usually, it needs to be converted to Tensor in Dataset.
        In the case of Text, since it is more efficient to tokenize a list of sentences together, 
        DataLoader converts them into Tensors.
        """
        super().__init__(dataset, dtype_data, dtype_target, **kwargs)
        self.tokenizer = tokenizer
        assert isinstance(tokenizer_params_input, dict)
        assert isinstance(tokenizer_params_label, dict)
        self.tokenizer_params_input = tokenizer_params_input
        self.tokenizer_params_label = tokenizer_params_label
        self.preprocs_input   = preprocs_input if isinstance(preprocs_input, list) else ([preprocs_input, ] if preprocs_input is not None else [])
        self.aftprocs_input   = aftprocs_input if isinstance(aftprocs_input, list) else ([aftprocs_input, ] if aftprocs_input is not None else [])
        self.preprocs_label   = preprocs_label if isinstance(preprocs_label, list) else ([preprocs_label, ] if preprocs_label is not None else [])
        self.aftprocs_label   = aftprocs_label if isinstance(aftprocs_label, list) else ([aftprocs_label, ] if aftprocs_label is not None else [])
    def collate_fn(self, batch):
        input, label = list(zip(*batch))
        input, label = list(input), list(label)
        for proc in self.preprocs_input: input = proc(input)
        for proc in self.preprocs_label: label = proc(label)
        input = self.tokenizer(input, **self.tokenizer_params_input)
        if check_type_list(label, str): label = self.tokenizer(label, **self.tokenizer_params_label)
        for proc in self.aftprocs_input: input = proc(input)
        for proc in self.aftprocs_label: label = proc(label)
        return self.to_tensor(input, label)


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
        data   = torch.from_numpy(data)
        target = torch.from_numpy(target)
        dataset = TensorDataset(data, target)
        super().__init__(dataset, dtype_data=dtype_data, dtype_target=dtype_target, **kwargs)
