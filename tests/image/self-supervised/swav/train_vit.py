import numpy as np
from PIL import Image 
import torch, kktorch
import kktorch.util.image as tfms
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import PASCALvoc2012DataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import SwAVLoss


class MyTrainer(Trainer):
    def process_data_train_pre(self, input):
        return [input, ] if isinstance(input, torch.Tensor) else input
    def process_data_valid_pre(self, input):
        return [input, ] if isinstance(input, torch.Tensor) else input
    def process_data_train_aft(self, input):
        if self.i_epoch <= 3:
            self.network.search_module("ParameterModule(cluster_param)").param.requires_grad = False
        return [input[-1], ]


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
        f"/{kktorch.__path__[0]}/model_zoo/swav/swav_vit.json", 
        user_parameters={
            "___n_layer": 6,
            "___n_dim": 256,
            "___n_head": 8,
            "___in_channels": 3,
            "___dropout_p": 0.0,
            "___patch_size": 16,
            "___img_size": 224,
            "___n_projection": 128,
            "___k_clusters": 128
        },
    )

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
        losses_train=SwAVLoss(temperature=0.1, sinkhorn_epsilon=0.05),
        losses_train_name="swav",
        optimizer={
            "optimizer": torch.optim.AdamW, 
            "params": dict(lr=1e-3)
        },
        dataloader_train=dataloader_train,
        epoch=100, print_step=200, auto_mixed_precision=True, accumulation_step=1, clip_grad=3.0
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()
