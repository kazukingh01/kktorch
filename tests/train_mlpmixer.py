import numpy as np
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

    # predict
    x, y = trainer.predict(trainer.dataloader_valids[0], is_label=True, sample_size=1)
    """
    >>> x
    array([[-2.20284894e-01, -2.81294918e+00, -2.01630354e+00,
            9.86757278e-01, -1.47188687e+00,  7.36571372e-01,
            -4.62583876e+00,  7.63404369e+00,  2.13403076e-01,
            2.61508441e+00],
            [ 4.46817040e-01, -1.84661615e+00,  4.16849852e+00,
            1.46590257e+00, -4.17302132e+00,  1.03368688e+00,
            1.59531510e+00, -3.58910871e+00,  7.30294049e-01,
            -3.84971356e+00],
            ...
            [-3.00343680e+00, -1.08394504e+00, -9.65300441e-01,
            -9.31822777e-01,  2.98482490e+00, -1.20111912e-01,
            -1.05130568e-01, -5.27108431e-01,  7.37953365e-01,
            3.82410455e+00],
            [-2.10396957e+00,  2.57409424e-01,  4.04717398e+00,
            3.46306038e+00, -2.76303744e+00, -1.31176692e-02,
            -4.58458453e-01, -3.30368090e+00,  2.34734106e+00,
            -3.39738679e+00]], dtype=float32)
    >>> y
    array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4, 9, 6,
        6, 5, 4, 0, 7, 4, 0, 1, 3, 1, 3, 4, 7, 2, 7, 1, 2, 1, 1, 7, 4, 2,
        3, 5, 1, 2, 4, 4, 6, 3, 5, 5, 6, 0, 4, 1, 9, 5, 7, 8, 9, 3])
    >>> (np.argmax(x, axis=1) == y).sum() / y.shape[0]
    0.9375
    """