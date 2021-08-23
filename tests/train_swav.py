import numpy as np
import torch, kktorch
from torchvision import transforms
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import PASCALvoc2012DataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import SwAVLoss
from kktorch.util.image.transforms import ResizeFixRatio

class MyTrainer(Trainer):
    def process_data_train_aft(self, input):
        if self.i_epoch % 2 == 1:
            self.network.swav[5].param.requires_grad = False
        else:
            self.network.swav[5].param.requires_grad = True
        return super().process_data_train_aft(input)

if __name__ == "__main__":
    # config file
    fjson = "../kktorch/model_zoo/swav/swav.json"

    # load config file and create network
    network = ConfigModule(
        fjson,
        ## You can override the config settings.
        user_parameters={
            "___n_projection": 128,
            "___k_clusters": 500,
            "___n_augmentation": 6
        },
    )

    # dataloader
    dataloader_train = PASCALvoc2012DataLoader(
        root='./data', train=True, download=True, batch_size=24, shuffle=True, drop_last=True,
        transforms=[
            transforms.Compose([
                ResizeFixRatio(256, "min"), transforms.ToTensor(), transforms.RandomCrop(256), 
                transforms.Normalize(
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
                )
            ]),
            transforms.Compose([
                ResizeFixRatio(256, "min"), transforms.ToTensor(), transforms.RandomCrop(256), 
                transforms.Normalize(
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
                )
            ]),
            transforms.Compose([
                ResizeFixRatio(256, "min"), transforms.ToTensor(), transforms.RandomCrop(128), 
                transforms.Normalize(
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
                )
            ]),
            transforms.Compose([
                ResizeFixRatio(256, "min"), transforms.ToTensor(), transforms.RandomCrop(128), 
                transforms.Normalize(
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
                )
            ]),
            transforms.Compose([
                ResizeFixRatio(256, "min"), transforms.ToTensor(), transforms.RandomCrop(128), 
                transforms.Normalize(
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
                )
            ]),
            transforms.Compose([
                ResizeFixRatio(256, "min"), transforms.ToTensor(), transforms.RandomCrop(128), 
                transforms.Normalize(
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
                )
            ]),
        ],
        num_workers=4
    )
    dataloader_valid = PASCALvoc2012DataLoader(
        root='./data', train=False, download=True, batch_size=1, shuffle=False, drop_last=False,
        transforms=[
            transforms.Compose([
                ResizeFixRatio(256, "min"), transforms.ToTensor(), transforms.RandomCrop(256), 
                transforms.Normalize(
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
                )
            ])
        ]
    )
    # trainer
    trainer = MyTrainer(
        network,
        losses_train=SwAVLoss(),
        losses_train_name="swav",
        optimizer={
            "optimizer": torch.optim.AdamW, 
            "params": dict(lr=0.01)
        }, 
        dataloader_train =dataloader_train,
        epoch=10, print_step=500, auto_mixed_precision=True
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # predict
    x, y = trainer.predict(dataloader_valid, is_label=False, sample_size=1)
