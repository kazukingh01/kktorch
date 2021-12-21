import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as tfms

import kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MVTecADDataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import IdentityLoss, SSIMLoss


class MyTrainer(Trainer):
    def process_label_pre(self, label, input):
        # return label as input for image reconstruction
        return [label, input]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type")
    args = parser.parse_args()
    assert args.type in ["ssim", "mse", "msenorm"]

    # network
    dim_z   = 128
    if args.type == "msenorm":
        fjson = f"/{kktorch.__path__[0]}/model_zoo/autoencoder/autoencoder.json"
    else:
        fjson = f"/{kktorch.__path__[0]}/model_zoo/autoencoder/autoencoder_hardtanh.json"
    network = ConfigModule(
        fjson, 
        user_parameters={
            "___in_channels": 3,
            "___init_dim": 8,
            "___n_layers_en": 6,
            "___n_layers_de": 6,
            "___z": dim_z,
            "___init_size": 2,
            "___alpha": 16
        }
    )

    # dataloader
    if args.type == "msenorm":
        transforms=tfms.Compose([
            tfms.Resize(128, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
            tfms.Normalize(
                MVTecADDataLoader.MVTecAD_DEFAULT_MEAN["capsule"],
                MVTecADDataLoader.MVTecAD_DEFAULT_STD[ "capsule"],
            )
        ])
    else:
        transforms=tfms.Compose([
            tfms.Resize(128, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
        ])
    dataloader_train = MVTecADDataLoader(
        datatype="capsule", train=True, download=True, batch_size=64, shuffle=True,
        transforms=transforms
    )
    dataloader_valid = MVTecADDataLoader(
        datatype="capsule", train=False, download=True, batch_size=64, shuffle=False,
        transforms=transforms
    )

    # trainer
    if args.type == "msenorm":
        losses_train=[IdentityLoss(), torch.nn.MSELoss()]
        losses_train_name=["ignore", "mse"]
        losses_valid=[IdentityLoss(), torch.nn.MSELoss()]
        losses_valid_name=["ignore", "mse"]
    elif args.type == "mse":
        losses_train=[IdentityLoss(), torch.nn.MSELoss()]
        losses_train_name=["ignore", "mse"]
        losses_valid=[IdentityLoss(), [torch.nn.MSELoss(), SSIMLoss(8, 3)]]
        losses_valid_name=["ignore", "mse", "ssim"]
    else:
        losses_train=[IdentityLoss(), SSIMLoss(8, 3)]
        losses_train_name=["ignore", "ssim"]
        losses_valid=[IdentityLoss(), [torch.nn.MSELoss(), SSIMLoss(8, 3)]]
        losses_valid_name=["ignore", "mse", "ssim"]
    trainer = MyTrainer(
        network,
        losses_train=losses_train, losses_train_name=losses_train_name,
        losses_valid=losses_valid, losses_valid_name=losses_valid_name,
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-4)},
        dataloader_train =dataloader_train,
        dataloader_valids=dataloader_valid,
        epoch=100, valid_step=1, valid_iter=-1, print_step=500, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()
