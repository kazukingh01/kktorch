# FFN ( Feedforward Neural Network )
Simple network

## Usage
```
>>> from kktorch.nn import ConfigModule
>>> network = ConfigModule("./ffn.json")
```

## Network
```
ConfigModule(
  (ffn): ModuleList(
    (0): Linear(in_features=28, out_features=512, bias=True)
    (1): GELU()
    (2): ConfigModule(
      (sub): ModuleList(
        (0): Linear(in_features=512, out_features=256, bias=True)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): GELU()
        (3): Dropout(p=0.2, inplace=False)
      )
    )
    (3): ConfigModule(
      (sub): ModuleList(
        (0): Linear(in_features=256, out_features=128, bias=True)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): GELU()
        (3): Dropout(p=0.2, inplace=False)
      )
    )
    (4): ConfigModule(
      (sub): ModuleList(
        (0): Linear(in_features=128, out_features=64, bias=True)
        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): GELU()
        (3): Dropout(p=0.2, inplace=False)
      )
    )
    (5): Linear(in_features=64, out_features=1, bias=True)
  )
)
```