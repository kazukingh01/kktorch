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
    print("predict train")
    x_train, y_train = trainer.predict(dataloader_train, is_label=True, sample_size=0)
    print("predict valid")
    x_valid, y_valid = trainer.predict(dataloader_valid, is_label=True, sample_size=0)
    x_train = x_train[0].mean(axis=1) # mean multi crop output
    x_valid = x_valid[0].mean(axis=1) # mean multi crop output

    # knn
    import faiss
    index = faiss.IndexFlatL2(x_train.shape[-1])
    index.add(x_train.astype(np.float32))
    D, I = index.search(x_valid.astype(np.float32), k=5)
    n = (y_train[I, :].mean(axis=1).argmax(axis=1) == y_valid.argmax(axis=1)).sum()
    print(len(dataloader_valid), n, f"acc: {n / len(dataloader_valid)}")

    # plot
    import cv2
    base_size = 200
    for i in range(1, 20):
        img = ResizeFixRatio(base_size, "height")(dataloader_valid.dataset.get_image(i))
        for j in I[i]: img = np.concatenate([img, ResizeFixRatio(base_size, "height")(dataloader_train.dataset.get_image(j))], axis=1) 
        cv2.imwrite(f"knn{i}.png", img)
