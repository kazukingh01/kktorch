import numpy as np
import cv2
import torch
from PIL import Image

from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MNISTDataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import VAE_KLDLoss
import torchvision.transforms as tfms


class MyTrainer(Trainer):
    def process_label_pre(self, label, input):
        return input


if __name__ == "__main__":
    # network
    dim_z = 32
    network = ConfigModule(
        "../kktorch/model_zoo/vae/vae.json", 
        user_parameters={"___z": dim_z}
    )

    # dataloader
    dataloader_train = MNISTDataLoader(
        root='./data', train=True,  download=True, batch_size=128, shuffle=True,
        transform=[
            tfms.ToTensor(), 
        ]
    )

    # trainer
    trainer = MyTrainer(
        network,
        losses_train=[VAE_KLDLoss(), torch.nn.BCELoss()],
        losses_train_name=["kld", "mse"],
        losses_train_weight=[0.01, 0.99],
        adjust_target_size_front=1,
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-4)}, 
        dataloader_train =dataloader_train, epoch=100, valid_step=10, print_step=100, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # test
    ndf, label = trainer.predict(dataloader=dataloader_train, is_label=True, sample_size=3)
    output = network.vae[1](torch.from_numpy(ndf[0][:, :dim_z]).to("cuda").to(torch.float32))
    output = (output.to("cpu").detach().numpy() * 255).astype(np.uint8)
    for i in range(output.shape[0]):
        cv2.imshow("test", output[i, 0])
        cv2.waitKey(0)

    output = (label[1] * 255).astype(np.uint8)
    for i in range(output.shape[0]):
        cv2.imshow("test", output[i, 0])
        cv2.waitKey(0)
