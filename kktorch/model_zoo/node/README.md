# NODE ( Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data )

## Paper
https://arxiv.org/abs/1909.06312

## Code for reference
https://github.com/Qwicen/node

## Usage
```
>>> from kktorch.nn import ConfigModule
>>> network = ConfigModule("./node.json")
```

## Network
```
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
                (0): ParameterModule(name=ParameterModule(param_select_features), dim=(28, 128, 4), init_type=rand, init_timing=before, requires_grad=True, dtype=torch.float32, output_type: parallel)
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
                (12): ParameterModule(name=ParameterModule(prob_distribution), dim=(128, 16, 2), init_type=randn, init_timing=before, requires_grad=True, dtype=torch.float32, output_type: parallel)
                (13): EinsumInput(einsum=btl,tlc->btc)
                (14): ReshapeInput(dim=(-1, 256))
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
                (0): ParameterModule(name=ParameterModule(param_select_features), dim=(284, 128, 4), init_type=rand, init_timing=before, requires_grad=True, dtype=torch.float32, output_type: parallel)
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
                (12): ParameterModule(name=ParameterModule(prob_distribution), dim=(128, 16, 2), init_type=randn, init_timing=before, requires_grad=True, dtype=torch.float32, output_type: parallel)
                (13): EinsumInput(einsum=btl,tlc->btc)
                (14): ReshapeInput(dim=(-1, 256))
              )
            )
          )
        )
        (2): CombineListInput(combine_type=cat, dim=-1)
      )
    )
    (2): SelectIndexTensorInput(max_dim=2, dim=[slice(None, None, None), slice(28, None, None)])
    (3): ReshapeInput(dim=(-1, 256, 2))
    (4): AggregateInput(aggregate=mean, kwargs={'dim': 1})
  )
)
```