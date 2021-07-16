import numpy as np
import torch
from torch import nn
from typing import List, Union

# local packages
import kktorch.util.tensor as util


__all__ = [
    "BaseModule",
    "MiddleSaveOutput",
    "MiddleReleaseOutput",
    "SplitOutput",
    "CombineListInput",
    "EinsumInput",
    "ReshapeInput",
    "SelectIndexListInput",
    "SelectIndexTensorInput",
    "ParameterModule",
    "AggregateInput",
    "EvalModule",
    "TimmModule",
]


class BaseModule(nn.Module):
    def __init__(self, name: str=None):
        super().__init__()
        self.name        = self.__class__.__name__ if name is None else name
        self.is_debug    = False
        self.save_output = None
    def forward(self, input: Union[torch.Tensor, List[torch.Tensor]]):
        output = input
        if self.is_debug:
            self.save_output = {self.name: output.clone() if isinstance(output, torch.Tensor) else output}
        return output


class MiddleSaveOutput(BaseModule):
    def __init__(self, name: str=None):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"))
        self.middle_output = None
    def forward(self, input: torch.Tensor):
        output = input
        self.middle_output = output.clone()
        return super().forward(output)


class MiddleReleaseOutput(BaseModule):
    def __init__(self, name: str=None):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"))
    def forward(self, input: torch.Tensor):
        output = input
        return super().forward(output)


class SplitOutput(BaseModule):
    def __init__(self, n_split: int, name: str=None):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"))
        assert isinstance(n_split, int) and n_split > 1
        self.n_split = n_split
    def extra_repr(self):
        return f'n_split={self.n_split}'
    def forward(self, input: torch.Tensor):
        output = [input] + [input.clone() for _ in range(self.n_split - 1)]
        return super().forward(output)


class CombineListInput(BaseModule):
    def __init__(self, combine_type: str, dim: int=-1, name: str=None):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"))
        assert isinstance(combine_type, str) and combine_type in ["sum", "ave", "cat"]
        self.is_check       = True
        self.dim            = dim
        self.forward_output = None
        self.combine_type   = combine_type
        if   combine_type == "sum": self.forward_output = self.forward_sum
        elif combine_type == "ave": self.forward_output = self.forward_ave
        elif combine_type == "cat": self.forward_output = self.forwart_cat
    def extra_repr(self):
        return f'combine_type={self.combine_type}, dim={self.dim}'
    def forward(self, input: List[torch.Tensor]) -> torch.Tensor:
        if self.is_check:
            # Check only the first time.
            assert isinstance(input, list) and len(input) > 0
            self.is_check = False
        output = self.forward_output(input)
        return super().forward(output)
    def forward_sum(self, input: List[torch.Tensor]) -> torch.Tensor:
        output = input[0]
        for x in input[1:]: output += x
        return output
    def forward_ave(self, input: List[torch.Tensor]) -> torch.Tensor:
        output = self.forward_sum(input)
        return output / len(input)
    def forwart_cat(self, input: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(input, dim=self.dim)


class SelectIndexListInput(BaseModule):
    def __init__(self, index: int, name: str=None):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"))
        self.index = index
    def forward(self, input: List[torch.Tensor]) -> torch.Tensor:
        output = input[self.index]
        return super().forward(output)


class EinsumInput(BaseModule):
    def __init__(self, einsum: str, name: str=None):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"))
        self.einsum = einsum
    def extra_repr(self):
        return f'einsum={self.einsum}'
    def forward(self, input: Union[torch.Tensor, List[torch.Tensor]]):
        output = input
        if not isinstance(input, list): output = [output]
        output = torch.einsum(self.einsum, *output)
        return super().forward(output)


class ReshapeInput(BaseModule):
    def __init__(self, *dim, name: str=None):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"))
        self.dim        = dim 
        self.is_not_set = True if sum([isinstance(x, str) for x in dim]) > 0 else False
    def extra_repr(self):
        return f'dim={self.dim}'
    def forward(self, input: torch.Tensor):
        if self.is_not_set:
            self.dim = self.convert_dim(self.dim, input.shape)
        output = input.reshape(*self.dim)
        return super().forward(output)
    @classmethod
    def convert_dim(cls, dim: list, shape: List[int]) -> List[int]:
        list_output = []
        dict_shape = {list("abcdefg")[i]:x for i, x in enumerate(shape)}
        for x in dim:
            if isinstance(x, str):
                list_output.append(eval(x, dict_shape))
            else: list_output.append(x)
        return list_output


class SelectIndexTensorInput(BaseModule):
    def __init__(
        self, max_dim: int, dim1=None, dim2=None, dim3=None, name: str=None, **kwargs):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"))
        assert max_dim >= 1 and max_dim < 7
        self.max_dim = max_dim
        self.dim1    = self.convert_dim(dim1)
        self.dim2    = self.convert_dim(dim2)
        self.dim3    = self.convert_dim(dim3)
        for x, y in kwargs.items(): setattr(self, x, self.convert_dim(y))
        self.forward_output = getattr(self, f"select_dim{max_dim}")
        for i in range(1, max_dim+1):
            if hasattr(self, f"dim{i}") == False:
                setattr(self, f"dim{i}", self.convert_dim(None))
    def extra_repr(self):
        return f'max_dim={self.max_dim}, dim={[getattr(self, "dim"+str(i)) for i in range(1, self.max_dim+1)]}'
    @classmethod
    def convert_dim(cls, dim: Union[int, str, List[int]]):
        if dim is None:
            return slice(None)
        elif isinstance(dim, int):
            return dim
        elif isinstance(dim, range):
            return list(dim)
        elif isinstance(dim, str):
            if dim.find(":") >= 0:
                return slice(*[int(x) if x != "None" else None for x in dim.split(":")])
            else:
                if dim.find("allpattern(") == 0:
                    x = int(eval(dim[len("allpattern("):-1], {}))
                    return torch.from_numpy(np.array([list(str(bin(i))[2:].zfill(int(np.log2(x - 1)) + 1)) for i in range(x)]).astype(int))
        elif isinstance(dim, list):
            return dim
    def select_dim1(self, input: torch.Tensor):
        return input[self.dim1]
    def select_dim2(self, input: torch.Tensor):
        return input[self.dim1, self.dim2]
    def select_dim3(self, input: torch.Tensor):
        return input[self.dim1, self.dim2, self.dim3]
    def select_dim4(self, input: torch.Tensor):
        return input[self.dim1, self.dim2, self.dim3, self.dim4]
    def select_dim5(self, input: torch.Tensor):
        return input[self.dim1, self.dim2, self.dim3, self.dim4, self.dim5]
    def forward(self, input: torch.Tensor):
        output = self.forward_output(input)
        return super().forward(output)


class ParameterModule(BaseModule):
    def __init__(
        self, *dim, 
        name: str="param", init_type: str="rand", init_timing: str="before", 
        requires_grad: bool=True, dtype=torch.float32, output_type: str="parallel", 
        eps: float=1e-6
    ):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"))
        assert not (init_timing == "after" and requires_grad)
        assert init_type in ["rand", "randn", "zero", "one", "custom", "eval"]
        self.dim            = dim
        self.init_type      = init_type
        self.init_timing    = init_timing
        self.requires_grad  = requires_grad
        self.dtype          = dtype
        self.is_not_set     = True if self.init_timing == "after" else False 
        if self.init_timing == "before": self.create_parameter(*self.dim)
        self.output_type    = output_type
        self.forward_output = lambda x, y: x
        if   output_type == "parallel": self.forward_output = lambda x, y: [x, y]
        elif output_type == "plus":     self.forward_output = lambda x, y: x + y
        elif output_type == "minus":    self.forward_output = lambda x, y: x - y
        elif output_type == "times":    self.forward_output = lambda x, y: x * y
        elif output_type == "divide":   self.forward_output = lambda x, y: x / (y + eps)
        elif output_type == "exp":      self.forward_output = lambda x, y: x * torch.exp(y + eps)
        elif output_type == "log":      self.forward_output = lambda x, y: x * torch.log(y + eps)
        else:                           self.forward_output = lambda x, y: eval(self.output_type, {"input": x, "param": y, "torch": torch})
    def extra_repr(self):
        return f'name={self.name}, dim={self.dim}, init_type={self.init_type}, init_timing={self.init_timing}, requires_grad={self.requires_grad}, dtype={self.dtype}, output_type: {self.output_type}'
    def forward(self, input: torch.Tensor):
        if self.is_not_set:
            self.create_parameter(*self.convert_dim(self.dim, input.shape), device=input.device)
            self.is_not_set = False
        output = self.forward_output(input, self.param)
        return super().forward(output)
    def create_parameter(self, *dim, device="cpu"):
        param = None
        if   self.init_type == "rand":   param = torch.rand( *dim)
        elif self.init_type == "randn":  param = torch.randn( *dim)
        elif self.init_type == "zero":   param = torch.zeros(*dim)
        elif self.init_type == "one":    param = torch.ones( *dim)
        elif self.init_type == "custom": param = torch.Tensor(dim)
        elif self.init_type == "eval":   param = eval(dim[0], {"util": util})
        self.param = nn.Parameter(param.to(self.dtype), requires_grad=self.requires_grad).to(device)
    @classmethod
    def convert_dim(cls, dim: list, shape: List[int]) -> List[int]:
        list_output = []
        for x in dim:
            if isinstance(x, str): list_output.append(shape[int(x)])
            else: list_output.append(x)
        return list_output


class AggregateInput(BaseModule):
    def __init__(self, aggregate: str, name: str=None, **kwargs):
        """
        Params::
            aggregate: torch function.
                ex) sum, prod, mean, ...
        """
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"))
        self.aggregate      = aggregate
        self.kwargs         = kwargs
        self.forward_output = getattr(torch, aggregate)
    def extra_repr(self):
        return f'aggregate={self.aggregate}, kwargs={self.kwargs}'
    def forward(self, input: torch.Tensor):
        output = self.forward_output(input, **self.kwargs)
        return super().forward(output)


class EvalModule(BaseModule):
    def __init__(self, eval: str, name: str=None):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"))
        self.eval = eval
    def extra_repr(self):
        return f'eval={self.eval}'
    def forward(self, input: torch.Tensor):
        output = eval(self.eval, {"input": input})
        return super().forward(output)


class PretrainedModule(BaseModule):
    def __init__(self, model: nn.Module, name_model: str, name_module: str=None, dict_freeze: dict=None):
        """
        Params::
            name_module: If string, You can use a module that is part of a model named "name_module"
            dict_freeze:
                ex) {"Linear": 10}, Freeze all modules until "Linear" is encountered 10 times.
        """
        import timm
        super().__init__(name=f"{self.__class__.__name__}({name_model})")
        self.name_model  = name_model
        self.dict_freeze = dict_freeze if dict_freeze is not None else {}
        self.model       = model if name_module is None else getattr(model, name_module)
        self.freeze()
    def extra_repr(self):
        return f'name_model={self.name_model}'
    def forward(self, input: torch.Tensor):
        output = self.model(input)
        return super().forward(output)
    def freeze(self):
        dictwk = {}
        for module in self.model.modules():
            name = module.__class__.__name__
            if dictwk.get(name) is None: dictwk[name] = 1
            else: dictwk[name] += 1
            for x, y in self.dict_freeze.items():
                if dictwk.get(x) is None or dictwk.get(x) <= y:
                    if hasattr(module, "weight") and module.weight is not None:
                        module.weight.requires_grad = False
                    if hasattr(module, "bias")   and module.bias   is not None:
                        module.bias.  requires_grad = False
        print(dictwk)


class TimmModule(PretrainedModule):
    def __init__(self, name_model: str, pretrained: bool=True, **kwargs):
        """
        Params::
            name_model: see: https://github.com/rwightman/pytorch-image-models/tree/master/timm/models
            pretrained: If true, load pretrained weight
        """
        import timm
        model = timm.create_model(name_model, pretrained=pretrained)
        super().__init__(model, name_model, **kwargs)


class HuggingfaceModule(PretrainedModule):
    def __init__(self, name_model: str, **kwargs):
        """
        Params::
            name_model: see: https://huggingface.co/models
        """
        from transformers import AutoTokenizer, AutoModel
        model = AutoModel.from_pretrained(name_model)
        super().__init__(model, name_model, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(name_model)
    def forward(self, input: dict):
        output = self.model(**input)
        return super().forward(output)
