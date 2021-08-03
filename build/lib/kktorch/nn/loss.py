import torch
from torch.nn.modules.loss import _Loss


__all__ = [
    "Accuracy",
    "CrossEntropyAcrossLoss",
    "CrossEntropySmoothingLoss",
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


class CrossEntropyAcrossLoss(torch.nn.CrossEntropyLoss):
        def __init__(self, embedding_dim: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.embedding_dim = embedding_dim
        def forward(self,input, target):
            return super().forward(input.reshape(-1, self.embedding_dim), target.reshape(-1))


class CrossEntropySmoothingLoss(_Loss):
    def __init__(self, classes: int, smoothing: float=0.0, reduction: str='mean', ignore_index: int=-100):
        assert isinstance(classes, int) and classes > 1
        assert isinstance(reduction, str) and reduction in ["mean", "sum"]
        super(CrossEntropySmoothingLoss, self).__init__(reduction=reduction)
        self.confidence   = 1.0 - smoothing
        self.smoothing    = smoothing
        self.classes      = classes
        self.ignore_index = ignore_index
        self.output_reduction = torch.mean if self.reduction == "mean" else torch.sum
        self.is_check = True
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.is_check:
            assert isinstance(input,  torch.Tensor) and len( input.shape) >= 2
            assert isinstance(target, torch.Tensor) and len(target.shape) == 1
            self.is_check = False
        input = input.log_softmax(dim=-1)
        true_dist = None
        with torch.no_grad():
            true_dist = torch.zeros_like(input, requires_grad=False).to(input.device)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        output = torch.sum(-true_dist * input, dim=-1)
        output = output[target != self.ignore_index]
        return self.output_reduction(output)