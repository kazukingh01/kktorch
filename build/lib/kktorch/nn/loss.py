from typing import List
import torch
from torch.nn.modules.loss import _Loss


__all__ = [
    "Accuracy",
    "CrossEntropyAcrossLoss",
    "CrossEntropySmoothingLoss",
    "SwAVLoss",
    "DINOLoss",
    "VAE_KLDLoss",
]


class BaseLoss(_Loss):
    def __init__(self, reduction="mean", is_check_everytime: bool=True):
        assert isinstance(reduction, str) and reduction in ["mean", "sum", "ident"]
        super().__init__()
        self.is_check = True
        self.is_check_everytime = is_check_everytime
        self.output_reduction   = lambda x: x
        if   self.reduction == "mean": self.output_reduction = torch.mean
        elif self.reduction == "sum":  self.output_reduction = torch.sum
    def forward(self, *args, **kwargs):
        if self.is_check_everytime or self.is_check:
            self.check(*args, **kwargs)
            self.is_check = False
        return self.output_reduction(self.forward_child(*args, **kwargs))
    def check(self): pass
    def forward_child(self):
        raise NotImplementedError


class Accuracy(BaseLoss):
    def __init__(self, threshold: float=0.5, is_check_everytime=False):
        super().__init__(reduction="mean", is_check_everytime=is_check_everytime)
        self.threshold = threshold
    def check(self, input: torch.Tensor, target: torch.Tensor):
        assert isinstance(input,  torch.Tensor)
        assert isinstance(target, torch.Tensor)
        if len(input.shape) == 1:
            assert len(target.shape) == 1
            assert input.shape[0] == target.shape[0]
        elif len(input.shape) == 2:
            assert target.dtype == torch.long
            assert len(target.shape) == 1
            assert input.shape[-1] >= torch.max(target)
    def forward_child(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 1:
            output = (target.to(torch.bool) == (input > self.threshold))
        elif len(input.shape) == 2:
            output = torch.max(input, dim=-1)[1]
            output = (output == target)
        output = output.to(torch.float32)
        return output


class CrossEntropyAcrossLoss(torch.nn.CrossEntropyLoss):
        def __init__(self, embedding_dim: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.embedding_dim = embedding_dim
        def forward(self,input, target):
            return super().forward(input.reshape(-1, self.embedding_dim), target.reshape(-1))


class CrossEntropySmoothingLoss(BaseLoss):
    def __init__(self, classes: int, smoothing: float=0.0, reduction: str='mean', ignore_index: int=-100, is_check_everytime=False):
        assert isinstance(classes, int) and classes > 1
        super().__init__(reduction=reduction, is_check_everytime=is_check_everytime)
        self.confidence   = 1.0 - smoothing
        self.smoothing    = smoothing
        self.classes      = classes
        self.ignore_index = ignore_index
    def check(self, input: torch.Tensor, target: torch.Tensor):
        assert isinstance(input,  torch.Tensor) and len( input.shape) >= 2
        assert isinstance(target, torch.Tensor) and len(target.shape) == 1
    def forward_child(self, input: torch.Tensor, target: torch.Tensor):
        input = input.log_softmax(dim=-1)
        true_dist = None
        with torch.no_grad():
            true_dist = torch.zeros_like(input, requires_grad=False).to(input.device)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        output = torch.sum(-true_dist * input, dim=-1)
        output = output[target != self.ignore_index]
        return output


class SwAVLoss(BaseLoss):
    """
    see: https://arxiv.org/abs/2006.09882
    """
    def __init__(
        self, temperature: float=0.1, sinkhorn_epsilon=0.05, sinkhorn_repeat: int=3,
        reduction: str='mean', is_check_everytime=False
    ):
        super().__init__(reduction=reduction, is_check_everytime=is_check_everytime)
        self.temperature      = temperature
        self.sinkhorn_repeat  = sinkhorn_repeat
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.log_softmax      = torch.nn.functional.log_softmax
    def check(self, input: torch.Tensor, *args):
        assert isinstance(input, torch.Tensor) and len(input.shape) == 3
    def forward_child(self, input: torch.Tensor, *args):
        """
        input: shape(Batch, N_Aug, K_cluster)
            # assume index starts from 1
            input[:,   1, :] ---> t1 Global Views
            input[:,   2, :] ---> t2 Global Views
            input[:,   3, :] ---> t3 Additional Small Views
            ...
            input[:, V+2, :] ---> tV+2 Additional Small Views
        """
        tens_zt = torch.einsum("abc->bac", input) # N_Aug, B_Batch, K_cluster
        with torch.no_grad():
            tens_zs  = tens_zt[:2].clone()
            tens_qs  = self.sinkhorn(tens_zs)
            tens_qs1 = tens_qs[0].expand(tens_zt.shape[0] - 1, -1, -1).detach()
            tens_qs2 = tens_qs[1].expand(tens_zt.shape[0] - 1, -1, -1).detach()
        tens_pt = self.log_softmax(tens_zt / self.temperature, dim=2)
        loss1   = (-1 * tens_qs1 * torch.cat([tens_pt[1:2], tens_pt[2:]], dim=0)).sum(dim=2)
        loss2   = (-1 * tens_qs2 * torch.cat([tens_pt[0:1], tens_pt[2:]], dim=0)).sum(dim=2)
        loss    = torch.cat([loss1.reshape(-1), loss2.reshape(-1)], dim=0)
        loss    = loss / (2 + 2 * (tens_zt.shape[0] - 2))
        return loss
    def sinkhorn(self, tens_zs: torch.Tensor):
        """
        tens_zs: shape(N_Aug, B_Batch, K_cluster)
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


class DINOLoss(BaseLoss):
    """
    see: https://arxiv.org/abs/2104.14294
    """
    def __init__(self, temperature_s: float=0.1, temperature_t: float=0.04, update_rate: float=0.9, reduction: str='mean', is_check_everytime=False):
        super().__init__(reduction=reduction, is_check_everytime=is_check_everytime)
        self.temperature_s = temperature_s
        self.temperature_t = temperature_t
        self.vec_center    = None
        self.update_rate   = update_rate
        self.softmax       = torch.nn.functional.softmax
        self.log_softmax   = torch.nn.functional.log_softmax
    def check(self, input: torch.Tensor, target: torch.Tensor):
        assert isinstance(input,  torch.Tensor) and len(input. shape) == 3
        assert isinstance(target, torch.Tensor) and len(target.shape) == 3
        assert input.shape == target.shape
        if self.vec_center is None:
            self.vec_center = torch.zeros(input.shape[-1], requires_grad=False).to(input.device)
    def forward_child(self, input: torch.Tensor, target: torch.Tensor):
        """
        input: List of torch.Tensor, [tensor_student: torch.Tensor, tensor_teacher: torch.Tensor]
            shape(B_Batch, N_Aug, D_Dimension)
            # assume index starts from 1
            input[:,   1, :] ---> t1 Global Views
            input[:,   2, :] ---> t2 Global Views
            input[:,   3, :] ---> t3 Additional Small Views
            ...
            input[:, V+2, :] ---> tV+2 Additional Small Views
        """
        input  = input.permute(1,0,2) # N_Aug, B_Batch, D_Dimension
        with torch.no_grad():
            target    = target.detach()[:, :2, :].permute(1,0,2) # N_Aug, B_Batch, D_Dimension
            output_t  = self.softmax((target - self.vec_center) / self.temperature_t, dim=-1)
            output_t1 = output_t[0].expand(input.shape[0] - 1, -1, -1)
            output_t2 = output_t[1].expand(input.shape[0] - 1, -1, -1)
        output_s = self.log_softmax(input / self.temperature_s, dim=-1)
        loss1    = (-1 * output_t1 * torch.cat([output_s[1:2], output_s[2:]], dim=0)).sum(dim=2)
        loss2    = (-1 * output_t2 * torch.cat([output_s[0:1], output_s[2:]], dim=0)).sum(dim=2)
        loss     = torch.cat([loss1.reshape(-1), loss2.reshape(-1)], dim=0)
        loss     = loss / (2 + 2 * (input.shape[0] - 2))
        with torch.no_grad():
            self.vec_center = self.vec_center.mul(self.update_rate) + target.mean(dim=(0,1)).mul(1 - self.update_rate)
        return loss


class VAE_KLDLoss(BaseLoss):
    def __init__(self, reduction: str='mean', is_check_everytime=False):
        super().__init__(reduction=reduction, is_check_everytime=is_check_everytime)
    def check(self, input: torch.Tensor, target):
        assert isinstance(input, torch.Tensor) and len(input.shape) == 2
    def forward_child(self, input: torch.Tensor, target):
        """
        input::
            gaussian parameters.
            shape is (batch_size, z_mean_dim + z_sigma_dim)
            z_sigma_dim: log(sigma^2)
        target::
            None
        """
        dim = input.shape[1] // 2
        input_z_mean, input_z_logsigma2 = input[:, :dim], input[:, dim:]
        loss = -0.5 * (1 + input_z_logsigma2 - input_z_mean ** 2 - torch.exp(input_z_logsigma2)).sum(axis=-1)
        return loss
