import torch, kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MNISTDataLoader
from kktorch.nn.configmod import ConfigModule


if __name__ == "__main__":
    # config file
    fjson = "../kktorch/model_zoo/mlpmixer/mlpmixer.json"

    # load config file and create network
    network = ConfigModule(
        fjson,
        ## You can override the config settings.
        user_parameters={
            "___pixel_h": 14,
            "___pixel_w": 14,
            "___n_rgb": 1,
            "___n_patch_h": 2,
            "___n_patch_w": 2,
            "___n_layers": 2,
            "___n_classes": 10
        },
    )
    """
    >>> network
    ConfigModule(
    (mlpmixer): ModuleList(
        (0): ReshapeInput(dim=(-1, 1, 28, 28))
        (1): ConfigModule(
        (divide_patch): ModuleList(
            (0): AggregateInput(aggregate=split, kwargs={'split_size_or_sections': 14, 'dim': 2})
            (1): EvalModule(eval=list(input))
            (2): ApplyModule(
            (apply_module): ConfigModule(
                (split): ModuleList(
                (0): AggregateInput(aggregate=split, kwargs={'split_size_or_sections': 14, 'dim': 3})
                (1): EvalModule(eval=list(input))
                )
            )
            )
            (3): EvalModule(eval=sum(input, []))
            (4): AggregateInput(aggregate=stack, kwargs={'dim': 0})
            (5): EinsumInput(einsum=pbchw->bpchw)
            (6): ReshapeInput(dim=(-1, 'b', 'c * d * e'))
        )
        )
        (2): Linear(in_features=196, out_features=196, bias=True)
        (3): RepeatModule(
        (list_module): ModuleList(
            (0): ConfigModule(
            (mixer_layer): ModuleList(
                (0): SkipConnection(
                (mlp1): ModuleList(
                    (0): LayerNorm((196,), eps=1e-05, elementwise_affine=True)
                    (1): EinsumInput(einsum=abc->acb)
                    (2): ConfigModule(
                    (mlp): ModuleList(
                        (0): Linear(in_features=4, out_features=4, bias=True)
                        (1): GELU()
                        (2): Linear(in_features=4, out_features=4, bias=True)
                    )
                    )
                    (3): EinsumInput(einsum=abc->acb)
                )
                )
                (1): SkipConnection(
                (mlp2): ModuleList(
                    (0): LayerNorm((196,), eps=1e-05, elementwise_affine=True)
                    (1): ConfigModule(
                    (mlp): ModuleList(
                        (0): Linear(in_features=196, out_features=196, bias=True)
                        (1): GELU()
                        (2): Linear(in_features=196, out_features=196, bias=True)
                    )
                    )
                )
                )
            )
            )
            (1): ConfigModule(
            (mixer_layer): ModuleList(
                (0): SkipConnection(
                (mlp1): ModuleList(
                    (0): LayerNorm((196,), eps=1e-05, elementwise_affine=True)
                    (1): EinsumInput(einsum=abc->acb)
                    (2): ConfigModule(
                    (mlp): ModuleList(
                        (0): Linear(in_features=4, out_features=4, bias=True)
                        (1): GELU()
                        (2): Linear(in_features=4, out_features=4, bias=True)
                    )
                    )
                    (3): EinsumInput(einsum=abc->acb)
                )
                )
                (1): SkipConnection(
                (mlp2): ModuleList(
                    (0): LayerNorm((196,), eps=1e-05, elementwise_affine=True)
                    (1): ConfigModule(
                    (mlp): ModuleList(
                        (0): Linear(in_features=196, out_features=196, bias=True)
                        (1): GELU()
                        (2): Linear(in_features=196, out_features=196, bias=True)
                    )
                    )
                )
                )
            )
            )
        )
        )
        (4): AggregateInput(aggregate=mean, kwargs={'dim': 1})
        (5): Linear(in_features=196, out_features=3, bias=True)
    )
    )
    """

    # trainer
    trainer = Trainer(
        network,
        losses_train=torch.nn.CrossEntropyLoss(),
        losses_valid=[[torch.nn.CrossEntropyLoss(), kktorch.nn.Accuracy()]],
        losses_train_name="ce",
        losses_valid_name=[["ce", "acc"]],
        optimizer={"optimizer": torch.optim.SGD, "params": dict(lr=0.001, weight_decay=0)}, 
        dataloader_train =MNISTDataLoader(root='./data', train=True,  download=True, batch_size=64, shuffle=True),
        dataloader_valids=MNISTDataLoader(root='./data', train=False, download=True, batch_size=64, shuffle=False),
        epoch=1000, valid_step=10, print_step=100, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()