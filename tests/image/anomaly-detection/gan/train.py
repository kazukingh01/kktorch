import numpy as np
import cv2
import torch
from PIL import Image

import kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MVTecADDataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import IdentityLoss, GANGeneratorLoss
import torchvision.transforms as tfms


class MyTrainer(Trainer):
    def aftproc_update_weight(self, input, label):
        #if self.iter % 2 == 0:
        loss = self.network.update_discriminator(label)
        loss = self.val_to_cpu(loss)
        self.write_tensor_board("train/gan_d", loss)


class GAN(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module, discriminator: torch.nn.Module):
        super().__init__()
        self.generator     = generator
        self.discriminator = discriminator
        self.loss_func     = torch.nn.BCELoss(reduction="mean")
        self.optimizerD    = torch.optim.AdamW(self.discriminator.parameters(), **dict(lr=1e-3, weight_decay=1e-2))
        self.param_for_device_check = torch.nn.Parameter(torch.ones(1))
    def forward(self, input: torch.nn.Module, *args, **kwargs):
        output = self.generator(input)
        return [self.discriminator(output), output]
    def parameters(self, recurse: bool=True):
        return self.generator.parameters(recurse=recurse)
    def update_discriminator(self, target: torch.nn.Module):
        self.discriminator.zero_grad()
        target   = target.to(self.param_for_device_check.device)
        output_r = self.discriminator(target)
        loss_r   = self.loss_func(output_r, torch.ones_like(output_r).to(output_r.device))
        output_f = self.discriminator(self.generator(target).detach())
        loss_f   = self.loss_func(output_f, torch.zeros_like(output_f).to(output_f.device))
        loss     = loss_r + loss_f
        loss.backward()
        self.optimizerD.step()
        return loss


if __name__ == "__main__":
    # network
    generator = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/gan/generator.json", 
        user_parameters={
            "___noize_dim": 8,
            "___in_channels": 3,
            "___init_dim": 8,
            "___n_layers": 4,
            "___init_size": 4,
            "___alpha": 64
        }
    )
    discriminator = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/gan/discriminator.json", 
        user_parameters={
            "___in_channels": 3,
            "___init_dim": 8,
            "___n_layers": 3,
        }
    )
    network = GAN(generator, discriminator)

    # dataloader
    transforms=tfms.Compose([
        tfms.Resize(64, interpolation=Image.BICUBIC),
        tfms.RandomResizedCrop(64, scale=(0.7, 1.0), interpolation=Image.BICUBIC),
        tfms.ToTensor(), 
        tfms.Normalize(
            MVTecADDataLoader.MVTecAD_DEFAULT_MEAN["capsule"],
            MVTecADDataLoader.MVTecAD_DEFAULT_STD[ "capsule"],
        )
    ])
    dataloader_train = MVTecADDataLoader(
        datatype="capsule", train=True, download=True, batch_size=8, shuffle=True,
        transforms=transforms, is_return_input_as_label=True
    )
    dataloader_valid = MVTecADDataLoader(
        datatype="capsule", train=False, download=True, batch_size=8, shuffle=False,
        transforms=transforms, is_return_input_as_label=False
    )

    # trainer
    trainer = MyTrainer(
        network,
        losses_train=GANGeneratorLoss(), losses_train_name=["gan_g"],
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-3, weight_decay=1e-2)},
        dataloader_train =dataloader_train, adjust_target_size_front=1,
        epoch=400, valid_step=10, print_step=50, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # preview
output = network.generator(torch.rand(10).to(network.param_for_device_check.device))
for i in range(3):
   output[:, i, :, :] = output[:, i, :, :] * MVTecADDataLoader.MVTecAD_DEFAULT_STD["capsule"][i] + MVTecADDataLoader.MVTecAD_DEFAULT_MEAN["capsule"][i]

output = (np.clip(output.to("cpu").detach().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
output = output.transpose(0, 2, 3, 1)
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