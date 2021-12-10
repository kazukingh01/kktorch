import numpy as np
import cv2
import torch
from PIL import Image

from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MNISTDataLoader
from kktorch.nn.configmod import ConfigModule
import torchvision.transforms as tfms


class MyTrainer(Trainer):
    def process_label_pre(self, label, input):
        return input


if __name__ == "__main__":
    # network
    dim_z = 64
    network = ConfigModule(
        "../kktorch/model_zoo/autoencoder/autoencoder.json", 
        user_parameters={"___z": dim_z}
    )

    # dataloader
    dataloader_train = MNISTDataLoader(
        root='./data', train=True,  download=True, batch_size=128, shuffle=True,
        transform=[
            tfms.Resize(112, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
        ]
    )
    dataloader_valid = MNISTDataLoader(
        root='./data', train=False,  download=True, batch_size=128, shuffle=False,
        transform=[
            tfms.Resize(112, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
        ]
    )

    # trainer
    trainer = MyTrainer(
        network,
        losses_train=torch.nn.MSELoss(), losses_train_name="mse",
        losses_valid=torch.nn.MSELoss(), losses_valid_name="mse",
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-4)}, 
        dataloader_train =dataloader_train, 
        dataloader_valids=dataloader_valid,
        epoch=2, valid_step=10, print_step=100, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # preview
    ndf, label = trainer.predict(dataloader=dataloader_valid, is_label=True, sample_size=1)
    output = (ndf * 255).astype(np.uint8)
    output = np.concatenate([output, (label * 255).astype(np.uint8)], axis=-1) 
    for i in range(output.shape[0]):
        cv2.imshow("test", output[i, 0])
        cv2.waitKey(0)
