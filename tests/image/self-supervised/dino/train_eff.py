import copy
from PIL import Image 
import torch, kktorch

import kktorch.util.image as tfms
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import PASCALvoc2012DataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import DINOLoss


class MyTrainer(Trainer):
    def process_data_train_pre(self, input):
        return [input, ] if isinstance(input, torch.Tensor) else input
    def process_data_valid_pre(self, input):
        return [input, ] if isinstance(input, torch.Tensor) else input
    def process_data_train_aft(self, input):
        return [input[0], ]
    def process_data_valid_aft(self, input):
        return [input[0], ]
    def process_label_aft(self, label, input):
        return [input[1], ]
    def aftproc_update_weight(self):
        self.network.update_teacher_weight()


class TeacherStudent(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, update_rate: float=0.9):
        super().__init__()
        self.student = copy.deepcopy(model)
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters(): p.requires_grad = False
        self.update_rate = update_rate
    def forward(self, input):
        output_s = self.student(input)
        with torch.no_grad():
            output_t = self.teacher(input)
        return output_s, output_t
    def parameters(self, recurse: bool = True):
        return self.student.parameters(recurse=recurse)
    def update_teacher_weight(self):
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data.mul_(self.update_rate).add_((1 - self.update_rate) * param_s.detach().data)


def aug_train(size: int, scale=(0.4, 1.0), p_blue: float=1.0, p_sol: float=0.0):
    return tfms.Compose([
        tfms.RandomResizedCrop(size, scale=scale, interpolation=Image.BICUBIC),
        tfms.RandomHorizontalFlip(p=0.5),
        tfms.RandomApply([tfms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
        tfms.RandomGrayscale(p=0.2),
        tfms.RandomApply([tfms.GaussianBlur(3)], p=p_blue),
        tfms.RandomSolarize(128.0, p=p_sol),
        tfms.ToTensor(),
        tfms.Normalize(
            PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
            PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
        ),
    ])


if __name__ == "__main__":
    # load config file and create network
    network = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/dino/dino_eff.json", 
        user_parameters={
            "___n_dim": 1280,
            "___n_projection": 128
        },
    )
    network = TeacherStudent(network, update_rate=0.996)

    # dataloader. multi crop 3 x 224x224, 6 x 96x96
    dataloader_train = PASCALvoc2012DataLoader(
        train=True, download=True, batch_size=32, shuffle=True, drop_last=True,
        transforms=[
            aug_train(224, scale=(0.4, 1.0),  p_blue=1.0, p_sol=0.0),
            aug_train(224, scale=(0.4, 1.0),  p_blue=0.5, p_sol=0.1),
            aug_train(224, scale=(0.4, 1.0),  p_blue=0.1, p_sol=0.2),
            aug_train( 96, scale=(0.05, 0.4), p_blue=0.5, p_sol=0.0),
            aug_train( 96, scale=(0.05, 0.4), p_blue=0.5, p_sol=0.0),
            aug_train( 96, scale=(0.05, 0.4), p_blue=0.5, p_sol=0.0),
            aug_train( 96, scale=(0.05, 0.4), p_blue=0.5, p_sol=0.0),
            aug_train( 96, scale=(0.05, 0.4), p_blue=0.5, p_sol=0.0),
            aug_train( 96, scale=(0.05, 0.4), p_blue=0.5, p_sol=0.0),
        ],
        num_workers=32
    )

    # trainer
    trainer = MyTrainer(
        network,
        losses_train=DINOLoss(temperature_s=0.1, temperature_t=0.04, update_rate=0.9, n_global_view=3),
        losses_train_name="dino",
        optimizer={
            "optimizer": torch.optim.AdamW, 
            "params": dict(lr=1e-3)
        }, 
        dataloader_train=dataloader_train,
        epoch=100, print_step=100, auto_mixed_precision=True, accumulation_step=1, clip_grad=3.0
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()
