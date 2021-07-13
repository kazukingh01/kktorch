# MLP-Mixer

## Paper
https://arxiv.org/abs/2105.01601

## Usage
```
>>> from kktorch.nn import ConfigModule
>>> network = ConfigModule("./mlpmixer.json")
```

## Network (mlpmixer.json)
```
ConfigModule(
  (mlpmixer): ModuleList(
    (0): ReshapeInput(dim=(-1, 3, 224, 224))
    (1): ConfigModule(
      (divide_patch): ModuleList(
        (0): AggregateInput(aggregate=split, kwargs={'split_size_or_sections': 16, 'dim': 2})
        (1): EvalModule(eval=list(input))
        (2): ApplyModule(
          (apply_module): ConfigModule(
            (split): ModuleList(
              (0): AggregateInput(aggregate=split, kwargs={'split_size_or_sections': 16, 'dim': 3})
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
    (2): Linear(in_features=768, out_features=768, bias=True)
    (3): RepeatModule(
      (list_module): ModuleList(
        (0): ConfigModule(
          (mixer_layer): ModuleList(
            (0): SkipConnection(
              (mlp1): ModuleList(
                (0): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (1): EinsumInput(einsum=abc->acb)
                (2): ConfigModule(
                  (mlp): ModuleList(
                    (0): Linear(in_features=196, out_features=196, bias=True)
                    (1): GELU()
                    (2): Linear(in_features=196, out_features=196, bias=True)
                  )
                )
                (3): EinsumInput(einsum=abc->acb)
              )
            )
            (1): SkipConnection(
              (mlp2): ModuleList(
                (0): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (1): ConfigModule(
                  (mlp): ModuleList(
                    (0): Linear(in_features=768, out_features=768, bias=True)
                    (1): GELU()
                    (2): Linear(in_features=768, out_features=768, bias=True)
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
                (0): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (1): EinsumInput(einsum=abc->acb)
                (2): ConfigModule(
                  (mlp): ModuleList(
                    (0): Linear(in_features=196, out_features=196, bias=True)
                    (1): GELU()
                    (2): Linear(in_features=196, out_features=196, bias=True)
                  )
                )
                (3): EinsumInput(einsum=abc->acb)
              )
            )
            (1): SkipConnection(
              (mlp2): ModuleList(
                (0): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (1): ConfigModule(
                  (mlp): ModuleList(
                    (0): Linear(in_features=768, out_features=768, bias=True)
                    (1): GELU()
                    (2): Linear(in_features=768, out_features=768, bias=True)
                  )
                )
              )
            )
          )
        )
      )
    )
    (4): AggregateInput(aggregate=mean, kwargs={'dim': 1})
    (5): Linear(in_features=768, out_features=3, bias=True)
  )
)
```

## Network (mlpmixer_rec.json)
Image Reconstruction ver.
```
ConfigModule(
  (mlpmixer_rec): ModuleList(
    (0): ReshapeInput(dim=(-1, 3, 224, 224))
    (1): ConfigModule(
      (divide_patch): ModuleList(
        (0): AggregateInput(aggregate=split, kwargs={'split_size_or_sections': 16, 'dim': 2})
        (1): EvalModule(eval=list(input))
        (2): ApplyModule(
          (apply_module): ConfigModule(
            (split): ModuleList(
              (0): AggregateInput(aggregate=split, kwargs={'split_size_or_sections': 16, 'dim': 3})
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
    (2): Linear(in_features=768, out_features=768, bias=True)
    (3): RepeatModule(
      (list_module): ModuleList(
        (0): ConfigModule(
          (mixer_layer): ModuleList(
            (0): SkipConnection(
              (mlp1): ModuleList(
                (0): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (1): EinsumInput(einsum=abc->acb)
                (2): ConfigModule(
                  (mlp): ModuleList(
                    (0): Linear(in_features=196, out_features=196, bias=True)
                    (1): GELU()
                    (2): Linear(in_features=196, out_features=196, bias=True)
                  )
                )
                (3): EinsumInput(einsum=abc->acb)
              )
            )
            (1): SkipConnection(
              (mlp2): ModuleList(
                (0): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (1): ConfigModule(
                  (mlp): ModuleList(
                    (0): Linear(in_features=768, out_features=768, bias=True)
                    (1): GELU()
                    (2): Linear(in_features=768, out_features=768, bias=True)
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
                (0): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (1): EinsumInput(einsum=abc->acb)
                (2): ConfigModule(
                  (mlp): ModuleList(
                    (0): Linear(in_features=196, out_features=196, bias=True)
                    (1): GELU()
                    (2): Linear(in_features=196, out_features=196, bias=True)
                  )
                )
                (3): EinsumInput(einsum=abc->acb)
              )
            )
            (1): SkipConnection(
              (mlp2): ModuleList(
                (0): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (1): ConfigModule(
                  (mlp): ModuleList(
                    (0): Linear(in_features=768, out_features=768, bias=True)
                    (1): GELU()
                    (2): Linear(in_features=768, out_features=768, bias=True)
                  )
                )
              )
            )
          )
        )
      )
    )
    (4): ConfigModule(
      (concat_patch): ModuleList(
        (0): ReshapeInput(dim=(-1, 'b', 3, 16, 16))
        (1): EinsumInput(einsum=bpchw->pbchw)
        (2): EvalModule(eval=[[input[14 * j + i] for i in range(14)] for j in range(14)])
        (3): ApplyModule(
          (apply_module): ConfigModule(
            (concat_w): ModuleList(
              (0): CombineListInput(combine_type=cat, dim=3)
            )
          )
        )
        (4): CombineListInput(combine_type=cat, dim=2)
      )
    )
  )
)
```
