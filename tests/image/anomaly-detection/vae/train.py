import numpy as np
import cv2
import torch
from PIL import Image

import kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MVTecADDataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import VAE_KLDLoss
import torchvision.transforms as tfms


class MyTrainer(Trainer):
    def process_label_pre(self, label, input):
        # return label as input for image reconstruction
        return [label, input]


if __name__ == "__main__":
    # network
    dim_z   = 128
    network = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/autoencoder/vae_norm.json", 
        user_parameters={
            "___in_channels": 3,
            "___init_dim": 8,
            "___n_layers_en": 6,
            "___n_layers_de": 5,
            "___z": dim_z,
            "___init_size": 4,
            "___alpha": 8
        }
    )

    # dataloader
    dataloader_train = MVTecADDataLoader(
        train=True, download=True, batch_size=32, shuffle=True,
        transforms=tfms.Compose([
            tfms.Resize(64, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
            tfms.Normalize(
                MVTecADDataLoader.MVTecAD_DEFAULT_MEAN, 
                MVTecADDataLoader.MVTecAD_DEFAULT_STD
            ),
        ])
    )
    dataloader_valid = MVTecADDataLoader(
        train=False, download=True, batch_size=32, shuffle=False,
        transforms=tfms.Compose([
            tfms.Resize(64, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
            tfms.Normalize(
                MVTecADDataLoader.MVTecAD_DEFAULT_MEAN, 
                MVTecADDataLoader.MVTecAD_DEFAULT_STD
            ),
        ])
    )

    # trainer
    trainer = MyTrainer(
        network,
        losses_train=[VAE_KLDLoss(), torch.nn.MSELoss()], losses_train_name=["kld", "mse"],
        losses_train_weight=[0.01, 1.0],
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-3, weight_decay=0)},
        dataloader_train =dataloader_train, 
        epoch=100, valid_step=10, print_step=500, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # preview
ndf, label = trainer.predict(dataloader=dataloader_valid, is_label=True, sample_size=0)
output = network.vae[3](torch.from_numpy(ndf[0][:, :dim_z]).to("cuda").to(torch.float32))
img_gt = label[1]
#for i in range(3):
#    output[:, i, :, :] = output[:, i, :, :] * MVTecADDataLoader.MVTecAD_DEFAULT_STD[i] + MVTecADDataLoader.MVTecAD_DEFAULT_MEAN[i]
#    img_gt[:, i, :, :] = img_gt[:, i, :, :] * MVTecADDataLoader.MVTecAD_DEFAULT_STD[i] + MVTecADDataLoader.MVTecAD_DEFAULT_MEAN[i]

output = (np.clip(output.to("cpu").detach().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
img_gt = (np.clip(img_gt, 0.0, 1.0) * 255).astype(np.uint8)
output = np.concatenate([img_gt, output], axis=-1)
output = output.transpose(0, 2, 3, 1)
img    = None
for i in range(16):
    cv2.imshow("test", output[i])
    cv2.waitKey(0)

"""    
for i in range(8):
    imgwk = np.concatenate([output[i*16+j, 0, :, :] for j in range(16)], axis=0)
    if img is None:
        img = imgwk
    else:
        img = np.concatenate([img, imgwk], axis=-1)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.imwrite("./imgrec.png", img)

    # t-SNE
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    ndf_z, _ = ndf
    ndf_z    = ndf_z[:, :ndf_z.shape[-1]//2]
    label_cls, _ = label
    tsne     = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000)
    ndf_tsne = tsne.fit_transform(ndf_z)
    colors   =  ["r", "g", "b", "c", "m", "y", "k", "orange","pink", "gray"]
    plt.figure(figsize = (6, 4))
    for i , v in enumerate(np.unique(label_cls)):
        ndf_bool = (label_cls == v)
        plt.scatter(ndf_tsne[ndf_bool, 0], ndf_tsne[ndf_bool, 1], label=v, color=colors[i], s=5, alpha=0.5)
    plt.legend()
    plt.savefig("t-sne.png")
    plt.show()
"""