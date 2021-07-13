import torch
from torch_optimizer import QHAdam
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import RandomDataLoader
from kktorch.nn.configmod import ConfigModule


if __name__ == "__main__":
    # input parameter
    in_features = 32
    n_classes   = 3

    # config file
    fjson = "../kktorch/model_zoo/node/node.json"

    # load config file and create network
    network = ConfigModule(
        fjson, in_features=in_features,
        ## You can override the config settings.
        user_parameters={
            "___in_features": in_features,
            "___n_trees": 128,
            "___depth": 4,
            "___n_classes": n_classes
        },
    )
    """
    >>> network
    ConfigModule(
    (node): ModuleList(
        (0): ConfigModule(
        (middle): ModuleList(
            (0): SplitOutput(n_split=2)
            (1): SplitModule(
            (list_module): ModuleList(
                (0): ConfigModule(
                (identity): ModuleList(
                    (0): Identity()
                )
                )
                (1): ConfigModule(
                (layer): ModuleList(
                    (0): ParameterModule(name=ParameterModule(param_select_features), dim=(32, 128, 4), init_type=rand, init_timing=before, requires_grad=True, dtype=torch.float32, output_type: parallel)
                    (1): SplitModule(
                    (list_module): ModuleList(
                        (0): ConfigModule(
                        (identity): ModuleList(
                            (0): Identity()
                        )
                        )
                        (1): ConfigModule(
                        (entmax): ModuleList(
                            (0): Entmax15()
                        )
                        )
                    )
                    )
                    (2): EinsumInput(einsum=ab,bcd->acd)
                    (3): ParameterModule(name=ParameterModule(param_threshold), dim=(128, 4), init_type=randn, init_timing=before, requires_grad=True, dtype=torch.float32, output_type: minus)
                    (4): ParameterModule(name=ParameterModule(param_scale), dim=(128, 4), init_type=randn, init_timing=before, requires_grad=True, dtype=torch.float32, output_type: input * torch.exp(param))
                    (5): Entmoid15()
                    (6): ReshapeInput(dim=(-1, 128, 4, 1))
                    (7): EvalModule(eval=[1 - input, input])
                    (8): CombineListInput(combine_type=cat, dim=-1)
                    (9): ParameterModule(name=ParameterModule(param_select_prob), dim=('util.create_allpattern(4)',), init_type=eval, init_timing=before, requires_grad=False, dtype=torch.float32, output_type: parallel)
                    (10): EinsumInput(einsum=btdp,dlp->btld)
                    (11): AggregateInput(aggregate=prod, kwargs={'dim': -1})
                    (12): ParameterModule(name=ParameterModule(prob_distribution), dim=(128, 16, 3), init_type=randn, init_timing=before, requires_grad=True, dtype=torch.float32, output_type: parallel)
                    (13): EinsumInput(einsum=btl,tlc->btc)
                    (14): ReshapeInput(dim=(-1, 384))
                )
                )
            )
            )
            (2): CombineListInput(combine_type=cat, dim=-1)
        )
        )
        (1): ConfigModule(
        (middle): ModuleList(
            (0): SplitOutput(n_split=2)
            (1): SplitModule(
            (list_module): ModuleList(
                (0): ConfigModule(
                (identity): ModuleList(
                    (0): Identity()
                )
                )
                (1): ConfigModule(
                (layer): ModuleList(
                    (0): ParameterModule(name=ParameterModule(param_select_features), dim=(416, 128, 4), init_type=rand, init_timing=before, requires_grad=True, dtype=torch.float32, output_type: parallel)
                    (1): SplitModule(
                    (list_module): ModuleList(
                        (0): ConfigModule(
                        (identity): ModuleList(
                            (0): Identity()
                        )
                        )
                        (1): ConfigModule(
                        (entmax): ModuleList(
                            (0): Entmax15()
                        )
                        )
                    )
                    )
                    (2): EinsumInput(einsum=ab,bcd->acd)
                    (3): ParameterModule(name=ParameterModule(param_threshold), dim=(128, 4), init_type=randn, init_timing=before, requires_grad=True, dtype=torch.float32, output_type: minus)
                    (4): ParameterModule(name=ParameterModule(param_scale), dim=(128, 4), init_type=randn, init_timing=before, requires_grad=True, dtype=torch.float32, output_type: input * torch.exp(param))
                    (5): Entmoid15()
                    (6): ReshapeInput(dim=(-1, 128, 4, 1))
                    (7): EvalModule(eval=[1 - input, input])
                    (8): CombineListInput(combine_type=cat, dim=-1)
                    (9): ParameterModule(name=ParameterModule(param_select_prob), dim=('util.create_allpattern(4)',), init_type=eval, init_timing=before, requires_grad=False, dtype=torch.float32, output_type: parallel)
                    (10): EinsumInput(einsum=btdp,dlp->btld)
                    (11): AggregateInput(aggregate=prod, kwargs={'dim': -1})
                    (12): ParameterModule(name=ParameterModule(prob_distribution), dim=(128, 16, 3), init_type=randn, init_timing=before, requires_grad=True, dtype=torch.float32, output_type: parallel)
                    (13): EinsumInput(einsum=btl,tlc->btc)
                    (14): ReshapeInput(dim=(-1, 384))
                )
                )
            )
            )
            (2): CombineListInput(combine_type=cat, dim=-1)
        )
        )
        (2): SelectIndexTensorInput(max_dim=2, dim=[slice(None, None, None), slice(32, None, None)])
        (3): ReshapeInput(dim=(-1, 256, 3))
        (4): AggregateInput(aggregate=mean, kwargs={'dim': 1})
    )
    )
    """

    # trainer
    trainer = Trainer(
        network,
        losses_train=torch.nn.CrossEntropyLoss(),
        losses_valid=torch.nn.CrossEntropyLoss(),
        losses_train_name="ce",
        losses_valid_name="ce",
        optimizer={"optimizer": QHAdam, "params": dict(nus=(0.7, 1.0), betas=(0.95, 0.998))}, 
        dataloader_train =RandomDataLoader(1024, in_features, n_classes, target_type="cls", batch_size=512, shuffle=True),
        dataloader_valids=RandomDataLoader(1024, in_features, n_classes, target_type="cls", batch_size=512, shuffle=False),
        epoch=1000, valid_step=10, print_step=100, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()