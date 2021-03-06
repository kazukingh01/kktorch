# kktorch
- This package is wrapper package for pytorch( https://github.com/pytorch/pytorch ).
- You can define NN (both Network and Forward pass) with config file.

## Installation
First, You will need to manually install pytorch for your machine environment (see: https://pytorch.org/get-started/locally/ )!!
ex)
```
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```
second,
```
pip install git+https://github.com/kazukingh01/kktorch.git@v1.0.0
```

## Model Zoo
https://github.com/kazukingh01/kktorch/tree/main/kktorch/model_zoo

## Training examples with using Model Zoo
code samples is in "https://github.com/kazukingh01/kktorch/tree/main/tests"

### Run test code
```
# change working directory.
mkdir work
cd ./work
git clone https://github.com/kazukingh01/kktorch.git
cd ./kktorch/tests/
python train_ffn.py
```

## Simple Usage
```
python
>>> import torch
>>> from kktorch.nn import ConfigModule
>>> network = ConfigModule({"name": "test", "network": [{"class": "Linear", "args": [128, 8]}, {"class": "ReLU"}]})
>>> network
ConfigModule(
    (test): ModuleList(
        (0): Linear(in_features=128, out_features=8, bias=True)
        (1): ReLU()
    )
)
>>> network(torch.rand(2, 128))
tensor([[0.0000, 0.0621, 0.0000, 0.4308, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0377, 0.0000, 0.1924, 0.0000, 0.0000, 0.0788, 0.0000]],
        grad_fn=<ReluBackward0>)
```