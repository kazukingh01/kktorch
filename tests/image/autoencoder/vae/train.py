import numpy as np
import cv2
import torch
from PIL import Image

import kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MNISTDataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import VAE_KLDLoss, SSIMLoss
import torchvision.transforms as tfms


class MyTrainer(Trainer):
    def process_label_pre(self, label, input):
        # return label as input for image reconstruction
        return [label, input]


if __name__ == "__main__":
    # network
    dim_z = 8
    network = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/autoencoder/vae.json", 
        user_parameters={
            "___in_channels": 1,
            "___init_dim": 8,
            "___n_layers_en": 5,
            "___n_layers_de": 3,
            "___z": dim_z,
            "___init_size": 4,
            "___alpha": 2
        }
    )

    # dataloader
    dataloader_train = MNISTDataLoader(
        train=True, download=True, batch_size=128, shuffle=True,
        transform=[
            tfms.Resize(32, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
        ]
    )
    dataloader_valid = MNISTDataLoader(
        train=False, download=True, batch_size=128, shuffle=False,
        transform=[
            tfms.Resize(32, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
        ]
    )

    # trainer
    trainer = MyTrainer(
        network,
        losses_train=[VAE_KLDLoss(), torch.nn.BCELoss()], losses_train_name=["kld", "bce"],
        losses_train_weight=[0.001, 1.0],
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-3, weight_decay=0)},
        dataloader_train =dataloader_train, 
        epoch=30, valid_step=10, print_step=500, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # test
    ndf, label = trainer.predict(dataloader=dataloader_train, is_label=True, sample_size=3)
    output = network.vae[3](torch.from_numpy(ndf[0][:, :dim_z]).to("cuda").to(torch.float32))
    output = torch.nn.Sigmoid()(output)
    output = (output.to("cpu").detach().numpy() * 255).astype(np.uint8)
    for i in range(output.shape[0]):
        cv2.imshow("test", output[i, 0])
        cv2.waitKey(0)

    output = (label[1] * 255).astype(np.uint8)
    for i in range(output.shape[0]):
        cv2.imshow("test", output[i, 0])
        cv2.waitKey(0)
