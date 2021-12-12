# MLP-Mixer
https://arxiv.org/abs/2105.01601

## How to use
```
python train.py
```

# Architexture
```python
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
        (6): ReshapeInput(dim=[-1, 4, 196])
      )
    )
    (2): Linear(in_features=196, out_features=196, bias=True)
    (3): RepeatModule(
      (list_module): ModuleList(
        (0): ConfigModule(
          (mixer_layer): ModuleList(
            (0): RegidualModule(
              (mod1): ConfigModule(
                (ident): ModuleList(
                  (0): Identity()
                )
              )
              (mod2): ConfigModule(
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
            )
            (1): RegidualModule(
              (mod1): ConfigModule(
                (ident): ModuleList(
                  (0): Identity()
                )
              )
              (mod2): ConfigModule(
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
        (1): ConfigModule(
          (mixer_layer): ModuleList(
            (0): RegidualModule(
              (mod1): ConfigModule(
                (ident): ModuleList(
                  (0): Identity()
                )
              )
              (mod2): ConfigModule(
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
            )
            (1): RegidualModule(
              (mod1): ConfigModule(
                (ident): ModuleList(
                  (0): Identity()
                )
              )
              (mod2): ConfigModule(
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
    )
    (4): AggregateInput(aggregate=mean, kwargs={'dim': 1})
    (5): Linear(in_features=196, out_features=10, bias=True)
  )
)
```