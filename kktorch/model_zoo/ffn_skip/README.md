# FFN with Skip Connection
Simple network

## Usage
```
>>> from kktorch.nn import ConfigModule
>>> network = ConfigModule("./ffn_skip.json")
```

## Network
```
ConfigModule(
  (ffn): ModuleList(
    (0): Linear(in_features=28, out_features=512, bias=True)
    (1): GELU()
    (2): RepeatModule(
      (list_module): ModuleList(
        (0): ConfigModule(
          (resblock): ModuleList(
            (0): SkipConnection(
              (resblock): ModuleList(
                (0): Linear(in_features=512, out_features=256, bias=True)
                (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): GELU()
                (3): Dropout(p=0.2, inplace=False)
                (4): Linear(in_features=256, out_features=512, bias=True)
              )
            )
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): GELU()
            (3): Dropout(p=0.2, inplace=False)
          )
        )
        (1): ConfigModule(
          (resblock): ModuleList(
            (0): SkipConnection(
              (resblock): ModuleList(
                (0): Linear(in_features=512, out_features=256, bias=True)
                (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): GELU()
                (3): Dropout(p=0.2, inplace=False)
                (4): Linear(in_features=256, out_features=512, bias=True)
              )
            )
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): GELU()
            (3): Dropout(p=0.2, inplace=False)
          )
        )
      )
    )
    (3): Linear(in_features=512, out_features=256, bias=True)
    (4): RepeatModule(
      (list_module): ModuleList(
        (0): ConfigModule(
          (resblock): ModuleList(
            (0): SkipConnection(
              (resblock): ModuleList(
                (0): Linear(in_features=256, out_features=128, bias=True)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): GELU()
                (3): Dropout(p=0.2, inplace=False)
                (4): Linear(in_features=128, out_features=256, bias=True)
              )
            )
            (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): GELU()
            (3): Dropout(p=0.2, inplace=False)
          )
        )
        (1): ConfigModule(
          (resblock): ModuleList(
            (0): SkipConnection(
              (resblock): ModuleList(
                (0): Linear(in_features=256, out_features=128, bias=True)
                (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): GELU()
                (3): Dropout(p=0.2, inplace=False)
                (4): Linear(in_features=128, out_features=256, bias=True)
              )
            )
            (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): GELU()
            (3): Dropout(p=0.2, inplace=False)
          )
        )
      )
    )
    (5): Linear(in_features=256, out_features=1, bias=True)
  )
)
```