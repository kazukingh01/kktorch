import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as tfms

import kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MVTecADDataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.util.image import concat_images


class MyTrainer(Trainer):
    def process_label_pre(self, label, input):
        # return label as input for image reconstruction
        return [label, input]


if __name__ == "__main__":
    # parser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight")
    parser.add_argument("--type")
    args = parser.parse_args()
    assert args.type in ["ssim", "mse", "msenorm"]
    args = parser.parse_args()

    # network
    dim_z   = 128
    fjson   = f"/{kktorch.__path__[0]}/model_zoo/autoencoder/autoencoder_hardtanh.json"
    if args.type == "msenorm":
        fjson = f"/{kktorch.__path__[0]}/model_zoo/autoencoder/autoencoder.json"
    network = ConfigModule(
        fjson, 
        user_parameters={
            "___in_channels": 3,
            "___init_dim": 8,
            "___n_layers_en": 6,
            "___n_layers_de": 6,
            "___z": dim_z,
            "___init_size": 2,
            "___alpha": 16
        }
    )

    # dataloader
    if args.type == "msenorm":
        transforms=tfms.Compose([
            tfms.Resize(128, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
            tfms.Normalize(
                MVTecADDataLoader.MVTecAD_DEFAULT_MEAN["capsule"],
                MVTecADDataLoader.MVTecAD_DEFAULT_STD[ "capsule"],
            )
        ])
    else:
        transforms=tfms.Compose([
            tfms.Resize(128, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
        ])
    dataloader_train = MVTecADDataLoader(
        datatype="capsule", train=True, download=True, batch_size=64, shuffle=False,
        transforms=transforms
    )
    dataloader_valid = MVTecADDataLoader(
        datatype="capsule", train=False, download=True, batch_size=64, shuffle=False,
        transforms=transforms
    )

    # trainer
    trainer = MyTrainer(network)
    trainer.load(args.weight)
    trainer.to_cuda()

    # preview ( left GT, right predict )    
    import matplotlib.pyplot as plt
    for dataloader, name in zip([dataloader_train, dataloader_valid], ["train", "test"]):
        output, label  = trainer.predict(dataloader=dataloader, is_label=True, sample_size=-1)
        ndf_z, ndf_img = output
        label_cls, label_img = label
        if args.type == "msenorm":
            for i in range(3):
                ndf_img[  :, i, :, :] = ndf_img[  :, i, :, :] * MVTecADDataLoader.MVTecAD_DEFAULT_STD["capsule"][i] + MVTecADDataLoader.MVTecAD_DEFAULT_MEAN["capsule"][i]
                label_img[:, i, :, :] = label_img[:, i, :, :] * MVTecADDataLoader.MVTecAD_DEFAULT_STD["capsule"][i] + MVTecADDataLoader.MVTecAD_DEFAULT_MEAN["capsule"][i]
        ndf_img   = (np.clip(ndf_img,   0, 1) * 255).astype(np.uint8)
        label_img = (np.clip(label_img, 0, 1) * 255).astype(np.uint8)
        ndf_diff = np.clip(np.abs(label_img.astype(float) -  ndf_img.astype(float)), 0, 255).astype(np.uint8)
        output   = np.concatenate([label_img, ndf_img, ndf_diff], axis=-1)
        output   = output.transpose(0, 2, 3, 1)
        img      = concat_images(output, (8, 4), resize=(64*3,64))
        cv2.imwrite(f"rec_diff_{args.type}_{name}.png", img)
        if name == "test":
            # histogram
            fig   = plt.figure(figsize=(6, 4))
            ax    = fig.add_subplot(1,1,1)
            ndf_diff = np.stack([cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY) for _img in ndf_diff.transpose(0, 2, 3, 1)])
            diffs = []
            diffs.append(ndf_diff[label_cls == 0].reshape(-1))
            diffs.append(ndf_diff[label_cls == 1].reshape(-1))
            ax.hist(diffs, label=["normal", "anomaly"], bins=50, log=True, stacked=False)
            fig.legend()
            fig.savefig(f"./hist_{args.type}.png")
            # histogram
            fig   = plt.figure(figsize=(6, 4))
            ax    = fig.add_subplot(1,1,1)
            diffs = (ndf_diff > 65).sum(-1).sum(-1)
            ax.hist([diffs[label_cls == 0], diffs[label_cls == 1]], label=["normal", "anomaly"], bins=np.unique(diffs).shape[0], log=True, stacked=False)
            fig.legend()
            fig.savefig(f"./hist_over65_{args.type}.png")
            # Confusion-matrix
            import pandas as pd
            df         = pd.DataFrame([diffs >=5, label_cls]).T.astype(int)
            df.columns = ["pred", "gt"]
            df["cnt"]  = 1
            print(pd.pivot_table(df, index="pred", columns="gt", values="cnt", aggfunc="count"))

