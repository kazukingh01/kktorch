import numpy as np
import cv2
import torch
import torchvision.transforms as tfms
from PIL import Image

import kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MNISTDataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import DeepSVDDLoss


def create_circle(C: np.ndarray, R: float, niter: int=100):
    n_dim    = C.shape[0]
    rand_dim = np.stack([np.random.permutation(np.arange(n_dim)) for _ in range(niter)])
    rand_pm  = np.random.randint(0, 2, (niter, n_dim))
    rand_pm[rand_pm == 0] = -1
    ndf      = []
    for i in range(niter):
        rand_vec = np.zeros(n_dim)
        _R = R
        for j in rand_dim[i][:-1]:
            point = np.random.rand(1)[0] * _R * rand_pm[i, j]
            rand_vec[j] = point
            _R = np.sqrt(_R ** 2 - point ** 2)
        j = rand_dim[i][-1]
        point = np.sqrt(R ** 2 - (rand_vec ** 2).sum())
        rand_vec[j] = point * rand_pm[i, j]
        ndf.append(rand_vec)
    return np.stack(ndf) + C


def plot(trainer, dataloader, loss, name="train"):
    # scatter plot
    output, label  = trainer.predict(dataloader=dataloader, is_label=True, sample_size=-1)
    import matplotlib.pyplot as plt
    colors =  ["r", "g", "b", "c", "m", "y", "k", "orange","pink", "gray"]
    plt.figure(figsize = (9, 6))
    for i , v in enumerate(np.unique(label)):
        ndf_bool = (label == v)
        plt.scatter(output[ndf_bool, 0], output[ndf_bool, 1], label=v, color=colors[i], s=5, alpha=0.2)
    ## plot circle
    C = loss.C.to("cpu").detach().numpy()
    R = loss.R.to("cpu").item()
    for ratio, color in zip([1.0, 1.25, 1.5], ["g", "b", "r"]):
        ndf = create_circle(C, R * ratio, niter=1000)
        plt.scatter(ndf[:, 0], ndf[:, 1], color=color, s=5, alpha=0.2, label=f"R({ratio})")
    plt.legend()
    plt.savefig(f"mnist_{name}_z.png")
    # out of circle train data
    R = loss.R.to("cpu").item()
    ratio    = 1.25 if name == "train" else 1.0
    ndf_bool = np.sqrt(((output - C) ** 2).sum(axis=-1)) > (R * ratio)
    imgs     = dataloader.dataset.data.numpy()[ndf_bool]
    img      = np.concatenate([np.concatenate([x for x in imgs[i*8:(i+1)*8]], axis=0) for i in range(8)], axis=-1)
    cv2.imwrite(f"./mnist_{name}_anomaly.png", img)
    ndf_bool = np.sqrt(((output - C) ** 2).sum(axis=-1)) < (R * 0.1)
    imgs     = dataloader.dataset.data.numpy()[ndf_bool]
    img      = np.concatenate([np.concatenate([x for x in imgs[i*8:(i+1)*8]], axis=0) for i in range(8)], axis=-1)
    cv2.imwrite(f"./mnist_{name}_normal.png", img)


class MyTrainer(Trainer):
    def process_label_pre(self, label, input):
        # return label as input for image reconstruction
        return input


if __name__ == "__main__":
    # network
    dim_z   = 2
    encoder = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/autoencoder/encoder.json", 
        user_parameters={
            "___in_channels": 1,
            "___init_dim": 8,
            "___n_layers": 4,
            "___z": dim_z,
        }
    )
    decoder = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/autoencoder/decoder.json", 
        user_parameters={
            "___in_channels": 1,
            "___init_size": 2,
            "___alpha": 32,
            "___n_layers": 4,
            "___z": dim_z,
        }
    )

    # dataloader
    dataloader_train = MNISTDataLoader(
        train=True, download=True, batch_size=256, shuffle=False,
        transforms=[
            tfms.Resize(32, interpolation=Image.BICUBIC),
            tfms.ToTensor()
        ]
    )
    dataloader_valid = MNISTDataLoader(
        train=False, download=True, batch_size=256, shuffle=False,
        transforms=[
            tfms.Resize(32, interpolation=Image.BICUBIC),
            tfms.ToTensor()
        ]
    )

    # Pre-training ( autoencoder ) 
    trainer = MyTrainer(
        torch.nn.Sequential(encoder, decoder),
        losses_train=torch.nn.MSELoss(), losses_train_name=["mse"],
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-4)},
        dataloader_train =dataloader_train,
        epoch=1, valid_step=10, print_step=500, 
    )
    # to cuda
    trainer.to_cuda()
    # training
    trainer.train()

    # Main-training ( DeepSVDD ) 
    loss    = DeepSVDDLoss(nu=0.05, n_update_C=(len(dataloader_train) // dataloader_train.batch_size))
    trainer = Trainer(
        encoder,
        losses_train=loss, losses_train_name=["svdd"],
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-4)},
        dataloader_train =dataloader_train,
        epoch=3, valid_step=10, print_step=500, 
    )
    # to cuda
    trainer.to_cuda()
    # training
    trainer.train()

    # plot
    plot(trainer, dataloader_train, loss, "train")
    plot(trainer, dataloader_valid, loss, "valid")
