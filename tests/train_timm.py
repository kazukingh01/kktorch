import torch, kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MNISTDataLoader
from kktorch.nn.configmod import ConfigModule
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


if __name__ == "__main__":
    # config file
    fjson = "../kktorch/model_zoo/timm/timm.json"

    # load config file and create network
    network = ConfigModule(
        fjson,
        ## You can override the config settings.
        user_parameters={
            "___n_classes": 10
        },
    )

    # dataloader
    dataloader_train = MNISTDataLoader(
        root='./data', train=True, download=True, batch_size=64, shuffle=True,
        transform=[
            transforms.ToTensor(), transforms.Resize([256,256]), transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ], num_workers=4
    )
    dataloader_valid = MNISTDataLoader(
        root='./data', train=False, download=True, batch_size=64, shuffle=False,
        transform=[
            transforms.ToTensor(), transforms.Resize([256,256]), transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ], num_workers=4
    )

    # trainer
    trainer = Trainer(
        network,
        losses_train=torch.nn.CrossEntropyLoss(),
        losses_valid=[[torch.nn.CrossEntropyLoss(), kktorch.nn.Accuracy()]],
        losses_train_name="ce",
        losses_valid_name=[["ce", "acc"]],
        optimizer={"optimizer": torch.optim.SGD, "params": dict(lr=0.001, weight_decay=0)}, 
        dataloader_train =dataloader_train,
        dataloader_valids=dataloader_valid,
        epoch=1000, valid_step=10, print_step=100, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()