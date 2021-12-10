import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as tfms

import kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MNISTDataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import IdentityLoss, SSIMLoss

class MyTrainer(Trainer):
    def process_label_pre(self, label, input):
        # return label as input for image reconstruction
        return [label, input]


if __name__ == "__main__":
    # network
    network = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/autoencoder/autoencoder.json", 
        user_parameters={
            "___in_channels": 1,
            "___init_dim": 8,
            "___n_layers_en": 4,
            "___n_layers_de": 4,
            "___z": 64,
            "___init_size": 4,
            "___alpha": 2
        }
    )

    # dataloader
    dataloader_train = MNISTDataLoader(
        train=True, download=True, batch_size=64, shuffle=True,
        transform=[
            tfms.Resize(64, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
        ]
    )
    dataloader_valid = MNISTDataLoader(
        train=False, download=True, batch_size=64, shuffle=False,
        transform=[
            tfms.Resize(64, interpolation=Image.BICUBIC),
            tfms.ToTensor(), 
        ]
    )

    # trainer
    trainer = MyTrainer(
        network,
        losses_train=[IdentityLoss(), torch.nn.MSELoss()], losses_train_name=["ignore", "mse"],
        losses_valid=[IdentityLoss(), [torch.nn.MSELoss(), SSIMLoss(8, 1)]], losses_valid_name=["ignore", "mse", "ssim"],
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-4)},
        dataloader_train =dataloader_train,
        dataloader_valids=dataloader_valid,
        epoch=1, valid_step=10, print_step=500, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # preview ( left GT, right predict )
    output, label  = trainer.predict(dataloader=dataloader_valid, is_label=True, sample_size=-1)
    ndf_z, ndf_img = output
    label_cls, label_img = label
    output = (ndf_img * 255).astype(np.uint8)
    output = np.concatenate([(label_img * 255).astype(np.uint8), output], axis=-1)[:, 0, :, :]
    img    = None
    for i in range(8):
        imgwk = np.concatenate([output[i] for i in range(i*8, (i+1)*8)], axis=0)
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
    tsne     = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000)
    ndf_tsne = tsne.fit_transform(ndf_z)
    colors   =  ["r", "g", "b", "c", "m", "y", "k", "orange","pink", "gray"]
    plt.figure(figsize = (6, 4))
    for i , v in enumerate(np.unique(label_cls)):
        ndf_bool = (label_cls == v)
        plt.scatter(ndf_tsne[ndf_bool, 0], ndf_tsne[ndf_bool, 1], label=v, color=colors[i], s=5, alpha=0.5)
    plt.savefig("t-sne.png")
    plt.show()
    