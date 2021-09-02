import numpy as np
import torch, kktorch
from torchvision import transforms
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import PASCALvoc2012DataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import SwAVLoss
from kktorch.util.image.transforms import ResizeFixRatio


class MyTrainer(Trainer):
    def process_data_train_pre(self, input):
        return [input, ] if isinstance(input, torch.Tensor) else input
    def process_data_valid_pre(self, input):
        return [input, ] if isinstance(input, torch.Tensor) else input
    def process_data_train_aft(self, input):
        if self.i_epoch <= 1:
            self.network.search_module("ParameterModule(cluster_param)").param.requires_grad = False
        return [input[-1], ]


if __name__ == "__main__":
    # config file
    fjson = "../kktorch/model_zoo/swav/swav.json"

    # load config file and create network
    n_projection = 32
    k_clusters   = 60
    network = ConfigModule(
        fjson,
        ## You can override the config settings.
        user_parameters={
            "___n_projection": n_projection,
            "___k_clusters": k_clusters,
        },
    )

    def aug_train(sizeA: int, sizeB: int):
        return transforms.Compose([
            ResizeFixRatio(sizeA, "min", is_check_everytime=False), 
            transforms.RandomCrop(sizeB),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ]),
            transforms.ToTensor(),
            transforms.Normalize(
                PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
                PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
            ),
        ])
    def aug_valid(sizeA: int, sizeB: int):
        return transforms.Compose([
            ResizeFixRatio(sizeA, "min", is_check_everytime=False), 
            transforms.RandomCrop(sizeB),
            transforms.ToTensor(),
            transforms.Normalize(
                PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
                PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
            ),
        ])

    # dataloader. multi crop 2 x 256x256, 4 x 128x128
    dataloader_train = PASCALvoc2012DataLoader(
        root='./data', train=True, download=True, batch_size=20, shuffle=True, drop_last=True,
        transforms=[
            aug_train(256, 256),
            aug_train(256, 256),
            aug_train(256, 128),
            aug_train(256, 128),
            aug_train(256, 128),
            aug_train(256, 128),
        ],
        num_workers=8
    )

    # trainer
    trainer = MyTrainer(
        network,
        losses_train=SwAVLoss(temperature=0.1, sinkhorn_epsilon=0.05),
        losses_train_name="swav",
        optimizer={
            "optimizer": torch.optim.AdamW, 
            "params": dict(lr=1e-2)
        }, 
        dataloader_train=dataloader_train,
        epoch=50, print_step=200, auto_mixed_precision=True, accumulation_step=1
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # evaluation setup. multi crop 2 x 256x256
    dataloader_train = PASCALvoc2012DataLoader(
        root='./data', train=True, download=True, batch_size=64, shuffle=False, drop_last=False, num_workers=8,
        transforms=[
            aug_valid(256, 256),
            aug_valid(256, 256),
        ],
    )
    dataloader_valid = PASCALvoc2012DataLoader(
        root='./data', train=False, download=True, batch_size=64, shuffle=False, drop_last=False, num_workers=8,
        transforms=[
            aug_valid(256, 256),
            aug_valid(256, 256),
        ]
    )
    trainer.load("/path/to/model/weight.pth")
    trainer.to_cuda()

    # predict
    x_train, y_train = trainer.predict(dataloader_train, is_label=True, sample_size=0)
    """
    >>> x_train[0]
    array([[[-0.53202677,  0.04835039, -0.00055272, ...,  0.09932363,
            0.45389313, -0.00763132],
            [-0.5514341 ,  0.06276986,  0.00674734, ...,  0.12609968,
            0.4591551 , -0.01237479]],
        ...,
        [[-0.14429946,  0.05713231,  0.00286166, ..., -0.02261793,
            0.1281438 ,  0.01483384],
            [-0.22170353,  0.05151182, -0.00398395, ...,  0.00851832,
            0.19584277,  0.02191174]]], dtype=float32)
    >>> x_train[0].shape
    (5717, 2, 32)
    """
    x_valid, y_valid = trainer.predict(dataloader_valid, is_label=True, sample_size=0)
    x_train = x_train[0].mean(axis=1) # mean multi crop output
    x_valid = x_valid[0].mean(axis=1) # mean multi crop output

    # knn
    import faiss
    index = faiss.IndexFlatL2(x_train.shape[-1])
    index.add(x_train.astype(np.float32))
    D, I = index.search(x_valid.astype(np.float32), k=5)
    """
    >>> D
    array([[0.00050521, 0.00052023, 0.00115323, 0.00134861, 0.00158119],
        [0.00081217, 0.00113809, 0.00161076, 0.00280142, 0.00340712],
        [0.00166631, 0.00363696, 0.00414395, 0.00453103, 0.00543332],
        ...,
        [0.0020014 , 0.00202489, 0.00245035, 0.00355589, 0.00363195],
        [0.00039721, 0.00060582, 0.00060976, 0.0007081 , 0.00074315],
        [0.00186503, 0.0020678 , 0.00261247, 0.00307369, 0.00347137]],
        dtype=float32)
    >>> I
    array([[ 707, 1862, 3145, 3080, 3209],
        [2751, 1747, 1371, 4199, 5249],
        [  96,  495, 3470, 4979, 2392],
        ...,
        [3671, 1526, 1756, 5363, 4205],
        [2506, 5055, 4014, 4424, 1412],
        [5066, 3910, 3026, 4915, 5102]])
    """
    n = (y_train[I, :].mean(axis=1).argmax(axis=1) == y_valid.argmax(axis=1)).sum()
    print(len(dataloader_valid), n, f"acc: {n / len(dataloader_valid)}")

    # plot
    import cv2
    base_size = 200
    for i in range(1, 20):
        img = ResizeFixRatio(base_size, "height")(dataloader_valid.dataset.get_image(i))
        for j in I[i]: img = np.concatenate([img, ResizeFixRatio(base_size, "height")(dataloader_train.dataset.get_image(j))], axis=1) 
        cv2.imwrite(f"knn{i}.png", img)