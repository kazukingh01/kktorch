import numpy as np
import torch
import torchvision.transforms as tfms
from PIL import Image

import kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MNISTDataLoader
from kktorch.nn.configmod import ConfigModule


if __name__ == "__main__":
    # load config file and create network
    network = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/vit/vit.json", 
        user_parameters={
            "___n_layer": 4,
            "___n_dim": 512,
            "___n_head": 8,
            "___in_channels": 1,
            "___dropout_p": 0.0,
            "___patch_size": 16,
            "___img_size": 32
        },
    )

    # dataloader
    dataloader_train = MNISTDataLoader(
        train=True, download=True, batch_size=128, shuffle=True,
        transforms=[
            tfms.Resize(32, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
        ]
    )
    dataloader_valid = MNISTDataLoader(
        train=False, download=True, batch_size=128, shuffle=False,
        transforms=[
            tfms.Resize(32, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
        ]
    )

    # trainer
    trainer = Trainer(
        network,
        losses_train=torch.nn.CrossEntropyLoss(),
        losses_valid=[[torch.nn.CrossEntropyLoss(), kktorch.nn.Accuracy()]],
        losses_train_name="ce",
        losses_valid_name=["ce", "acc"],
        optimizer={"optimizer": torch.optim.SGD, "params": dict(lr=0.01, weight_decay=0)}, 
        dataloader_train =dataloader_train, dataloader_valids=dataloader_valid,
        epoch=3, valid_step=10, print_step=100, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # predict
    x, y = trainer.predict(dataloader=dataloader_valid, is_label=True, sample_size=-1)
    acc  = (np.argmax(x, axis=1) == y).sum() / y.shape[0]
    print(f"acc: {acc}")
