import torch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import RandomDataLoader
from kktorch.nn.configmod import ConfigModule


if __name__ == "__main__":
    # input parameter
    in_features = 64
    n_classes   = 1

    # config file
    fjson = "../kktorch/model_zoo/ffn_skip/ffn_skip.json"

    # load config file and create network
    network = ConfigModule(
        fjson, in_features=in_features,
        ## You can override the config settings.
        user_parameters={
            "___p": 0.1,
            "___n_classes": n_classes,
            "___in_node": 256,
            "___n_layers": 2
        },
    )
    """
    >>> network
    ConfigModule(
        (ffn): ModuleList(
            (0): Linear(in_features=64, out_features=256, bias=True)
            (1): GELU()
            (2): RepeatModule(
            (list_module): ModuleList(
                (0): ConfigModule(
                (resblock): ModuleList(
                    (0): SkipConnection(
                    (resblock): ModuleList(
                        (0): Linear(in_features=256, out_features=128, bias=True)
                        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (2): GELU()
                        (3): Dropout(p=0.1, inplace=False)
                        (4): Linear(in_features=128, out_features=256, bias=True)
                    )
                    )
                    (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): GELU()
                    (3): Dropout(p=0.1, inplace=False)
                )
                )
                (1): ConfigModule(
                (resblock): ModuleList(
                    (0): SkipConnection(
                    (resblock): ModuleList(
                        (0): Linear(in_features=256, out_features=128, bias=True)
                        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (2): GELU()
                        (3): Dropout(p=0.1, inplace=False)
                        (4): Linear(in_features=128, out_features=256, bias=True)
                    )
                    )
                    (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): GELU()
                    (3): Dropout(p=0.1, inplace=False)
                )
                )
            )
            )
            (3): Linear(in_features=256, out_features=128, bias=True)
            (4): RepeatModule(
            (list_module): ModuleList(
                (0): ConfigModule(
                (resblock): ModuleList(
                    (0): SkipConnection(
                    (resblock): ModuleList(
                        (0): Linear(in_features=128, out_features=64, bias=True)
                        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (2): GELU()
                        (3): Dropout(p=0.1, inplace=False)
                        (4): Linear(in_features=64, out_features=128, bias=True)
                    )
                    )
                    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): GELU()
                    (3): Dropout(p=0.1, inplace=False)
                )
                )
                (1): ConfigModule(
                (resblock): ModuleList(
                    (0): SkipConnection(
                    (resblock): ModuleList(
                        (0): Linear(in_features=128, out_features=64, bias=True)
                        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (2): GELU()
                        (3): Dropout(p=0.1, inplace=False)
                        (4): Linear(in_features=64, out_features=128, bias=True)
                    )
                    )
                    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): GELU()
                    (3): Dropout(p=0.1, inplace=False)
                )
                )
            )
            )
            (5): Linear(in_features=128, out_features=1, bias=True)
        )
    )
    """

    # trainer
    trainer = Trainer(
        network,
        losses_train=torch.nn.MSELoss(),
        losses_valid=torch.nn.MSELoss(),
        losses_train_name="mse",
        losses_valid_name="mse",
        optimizer={"optimizer": torch.optim.SGD, "params": dict(lr=0.001, weight_decay=0)}, 
        scheduler={"scheduler": torch.optim.lr_scheduler.StepLR, "params": dict(step_size=1000, gamma=0.8)},
        dataloader_train =RandomDataLoader(1024, in_features, n_classes, target_type="reg", batch_size=512, shuffle=True),
        dataloader_valids=RandomDataLoader(1024, in_features, n_classes, target_type="reg", batch_size=512, shuffle=False),
        epoch=1000, valid_step=10, print_step=100, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # predict
    x, y = trainer.predict(trainer.dataloader_valids[0], is_label=True, sample_size=1)
    """
    >>> x
    array([[ 0.6218179 ],
        [ 0.4697637 ],
        [ 0.51726776],
        [ 0.17261215],
        ...
        [-0.15811712],
        [ 0.74395955],
        [ 0.681041  ],
        [ 0.47120208]], dtype=float32)
    >>> y
    array([[8.73586297e-01],
        [7.66591132e-01],
        [6.43652678e-03],
        [1.59346223e-01],
        ...
        [7.22083092e-01],
        [5.41172445e-01],
        [4.30119216e-01],
        [2.59101570e-01]], dtype=float32)
    """