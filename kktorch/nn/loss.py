import torch
from torch.nn.modules.loss import _Loss


__all__ = [
    "Accuracy",
    "CrossEntropyAcrossLoss",
    "CrossEntropySmoothingLoss",
    "SwAVLoss",
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


class SwAVLoss(_Loss):
    def __init__(
        self, temperature: float=0.1, 
        sinkhorn_repeat: int=3, sinkhorn_epsilon: float=0.05, 
        reduction: str='mean'
    ):
        super().__init__()
        self.temperature      = temperature
        self.sinkhorn_repeat  = sinkhorn_repeat
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.log_softmax      = torch.nn.functional.log_softmax
        self.reduction        = reduction
        self.output_reduction = torch.mean if self.reduction == "mean" else torch.sum
    def forward(self, input: torch.Tensor, *args):
        """
        input: shape(Batch, N_Aug, K_cluster)
            # index starts from 1
            input[:,   1, :] ---> t1 Global Views
            input[:,   2, :] ---> t2 Global Views
            input[:,   3, :] ---> t3 Additional Small Views
            ...
            input[:, V+2, :] ---> tV+2 Additional Small Views
        """
        tens_zt = torch.einsum("abc->bac", input) # N_Aug, Batch, K_cluster
        with torch.no_grad():
            tens_zs  = tens_zt[:2].clone()
            tens_qs  = self.sinkhorn(tens_zs)
            tens_qs1 = tens_qs[0].repeat(tens_zt.shape[0] - 1, 1, 1).detach()
            tens_qs2 = tens_qs[1].repeat(tens_zt.shape[0] - 1, 1, 1).detach()
        tens_pt = self.log_softmax(tens_zt / self.temperature, dim=2)
        loss1   = (-1 * tens_qs1 * torch.cat([tens_pt[1:2], tens_pt[2:]], dim=0)).sum(dim=2)
        loss2   = (-1 * tens_qs2 * torch.cat([tens_pt[0:1], tens_pt[2:]], dim=0)).sum(dim=2)
        loss    = torch.cat([loss1.reshape(-1), loss2.reshape(-1)], dim=0)
        return self.output_reduction(loss) / (2 * (tens_zt.shape[0] - 2))
    def sinkhorn(self, tens_zs: torch.Tensor):
        """
        tens_zs: shape(N_Aug, Batch, K_cluster)
        """
        Q = torch.exp(tens_zs / self.sinkhorn_epsilon).T
        Q = Q / torch.sum(Q, dim=(0,1))
        K, B, N = Q.shape
        u, r, c = torch.zeros(K, N), torch.ones(K, N) / K, torch.ones(B, N) / B
        u, r, c = u.to(tens_zs.device), r.to(tens_zs.device), c.to(tens_zs.device)
        for _ in range(self.sinkhorn_repeat):
            u = torch.sum(Q, dim=1)
            Q = Q * (r / u).unsqueeze(1)
            Q = Q * (c / torch.sum(Q, dim=0)).unsqueeze(0)
        Q = Q / torch.sum(Q, dim=0, keepdim=True)
        return Q.T