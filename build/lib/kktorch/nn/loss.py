import torch
from torch.nn.modules.loss import _Loss


__all__ = [
    "Accuracy"
]


class Accuracy(_Loss):
    def __init__(self, threshold: float=0.5):
        super().__init__()
        self.threshold = threshold
        self.is_check = True
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.is_check:
            assert isinstance(input,  torch.Tensor)
            assert isinstance(target, torch.Tensor)
            if len(input.shape) == 1:
                assert len(target.shape) == 1
                assert input.shape[0] == target.shape[0]
            elif len(input.shape) == 2:
                assert target.dtype == torch.long
                assert len(target.shape) == 1
                assert input.shape[-1] >= torch.max(target)
            self.is_check = False
        if len(input.shape) == 1:
            output = target.to(torch.bool) == (input > self.threshold)
        elif len(input.shape) == 2:
            output = torch.max(input, dim=-1)[1]
            output = output == target
        return torch.sum(output) / target.shape[0]