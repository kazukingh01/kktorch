import torch, kktorch
from torch_optimizer import QHAdam
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import RandomDataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import Accuracy


if __name__ == "__main__":
    # input parameter
    in_features = 32
    n_classes   = 3
    # load config file and create network
    network = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/node/node.json", 
        in_features=in_features,
        user_parameters={
            "___in_features": in_features,
            "___n_trees": 128,
            "___depth": 4,
            "___n_classes": n_classes
        },
    )

    # dataloader
    dataloader_train = RandomDataLoader(1024, in_features, n_classes, target_type="cls", batch_size=512, shuffle=True)
    dataloader_valid = RandomDataLoader(1024, in_features, n_classes, target_type="cls", batch_size=512, shuffle=False)

    # trainer
    trainer = Trainer(
        network,
        losses_train=torch.nn.CrossEntropyLoss(),
        losses_valid=[[torch.nn.CrossEntropyLoss(), Accuracy()]],
        losses_train_name="ce",
        losses_valid_name=["ce", "acc"],
        optimizer={"optimizer": QHAdam, "params": dict(nus=(0.7, 1.0), betas=(0.95, 0.998))}, 
        dataloader_train=dataloader_train, dataloader_valids=[dataloader_train, dataloader_valid],
        epoch=3000, valid_step=10, print_step=100, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # predict
    x, y = trainer.predict(trainer.dataloader_valids[0], is_label=True, sample_size=1)
    """
    >>> x
    array([[-0.6613625 ,  0.514653  , -0.10830054],
        [-0.0205371 , -0.11564586,  0.03547875],
        [ 0.9497983 , -0.6761255 , -0.48293447],
        ...,
        [ 0.02022172, -0.09065549, -0.07330489],
        [-0.08709136,  0.53957796, -0.5525283 ],
        [-0.52451336,  0.20215775,  0.2234598 ]], dtype=float32)
    >>> y
    array([2, 0, 1, 1, 2, 0, 0, 2, 1, 2, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 2, 2,
        2, 1, 2, 0, 0, 1, 2, 0, 1, 1, 2, 0, 0, 2, 2, 1, 2, 0, 2, 1, 1, 1,
        2, 2, 0, 0, 2, 1, 2, 0, 1, 0, 2, 1, 0, 0, 1, 1, 0, 2, 2, 2, 0, 1,
        ...,
        1, 1, 2, 0, 2, 2, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 2, 2, 1, 2, 1, 0,
        2, 0, 1, 0, 0, 2, 2, 0, 1, 2, 2, 0, 0, 2, 2, 1, 1, 1, 0, 2, 1, 1,
        2, 0, 2, 2, 2, 1])
    """
