import numpy as np
import torch, kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import PASCALvoc2012DataLoader
from kktorch.nn.configmod import ConfigModule
import kktorch.util.image as tfms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


if __name__ == "__main__":
    # load config file and create network
    network = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/timm/timm.json", 
        ## You can override the config settings.
        user_parameters={
            "___n_classes": 20,
            "___freeze_layers": [
                "^conv_stem\\.",
                "^bn1\\.",
                "^blocks\\.[0-5]\\.",
            ]
        },
    )

    # dataloader
    dataloader_train = PASCALvoc2012DataLoader(
        train=True, download=True, batch_size=32, shuffle=True,
        transforms=tfms.Compose([
            tfms.ToTensor(), tfms.ResizeFixRatio(256, "min"), tfms.RandomCrop(256),
            tfms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]), is_label_two_class=True, dtype_target=torch.float32, num_workers=4
    )
    dataloader_valid = PASCALvoc2012DataLoader(
        train=False, download=True, batch_size=32, shuffle=False,
        transforms=tfms.Compose([
            tfms.ToTensor(), tfms.ResizeFixRatio(256, "min"), tfms.RandomCrop(256),
            tfms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]), is_label_two_class=True, dtype_target=torch.float32, num_workers=4
    )

    # trainer
    trainer = Trainer(
        network,
        losses_train=torch.nn.BCELoss(),
        losses_valid=torch.nn.BCELoss(),
        losses_train_name="ce",
        losses_valid_name="ce",
        optimizer={"optimizer": torch.optim.SGD, "params": dict(lr=0.001, weight_decay=0)}, 
        dataloader_train =dataloader_train,
        dataloader_valids=dataloader_valid,
        epoch=10, valid_step=100, print_step=500, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # predict
    x, y = trainer.predict(dataloader_valid, is_label=True, sample_size=-1)
    acc  = (np.argmax(x, -1) == np.argmax(y, -1)).sum() / y.shape[0]
    print(f"acc: {acc}") # acc: 0.38073158165893867