from kktorch.util.image.numpy import pil2cv
import numpy as np
import cv2
import torch
import torchvision.transforms as tfms

import kktorch
import kktorch.util.image as tfms
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import PASCALvoc2012DataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.util.image import pil2cv


if __name__ == "__main__":
    # parameter
    img_size = 512
    # load config file and create network
    network = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/vit/vit_classification.json", 
        user_parameters={
            "___n_layer": 8,
            "___n_dim": 256,
            "___n_head": 8,
            "___in_channels": 3,
            "___dropout_p": 0.0,
            "___patch_size": 16,
            "___img_size": img_size,
            "___n_classes": 20
        },
    )

    # dataloader
    dataloader_train = PASCALvoc2012DataLoader(
        train=True, download=True, batch_size=16, shuffle=True,
        transforms=tfms.Compose([
            tfms.ToTensor(), tfms.ResizeFixRatio(img_size, "max"), tfms.CenterCrop(img_size),
            tfms.Normalize(PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD),
        ]), is_label_binary_class=True, dtype_target=torch.float32, num_workers=16
    )
    dataloader_valid = PASCALvoc2012DataLoader(
        train=False, download=True, batch_size=16, shuffle=False,
        transforms=tfms.Compose([
            tfms.ToTensor(), tfms.ResizeFixRatio(img_size, "max"), tfms.CenterCrop(img_size),
            tfms.Normalize(PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD),
        ]), is_label_binary_class=True, dtype_target=torch.float32, num_workers=16
    )

    # trainer
    trainer = Trainer(
        network,
        losses_train=torch.nn.BCELoss(), losses_valid=torch.nn.BCELoss(),
        losses_train_name="bce", losses_valid_name="bce",
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-4)}, 
        dataloader_train =dataloader_train, dataloader_valids=dataloader_valid,
        auto_mixed_precision=False, epoch=10, valid_step=200, valid_iter=10, print_step=100, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # predict
    x, y = trainer.predict(dataloader=dataloader_valid, is_label=True, sample_size=-1)
    acc  = (x[y.astype(bool)] > 0.3).sum() / y.astype(bool).sum() # threshold > 0.2
    print(f"acc: {acc}")

    # attention map
    result = []
    for I in range(4):
        resultwk = None
        for J in range(8):
            img      = dataloader_valid.dataset.get_image(I*8+J, "pil")
            img      = tfms.CenterCrop(img_size)(tfms.ResizeFixRatio(img_size, "max")(img))
            input, _ = dataloader_valid[I*8+J]
            output   = network.forward_debug(input.to("cuda"))
            attnmat  = [z[1].to("cpu") for x in output for y, z in x.items() if y == "selfattention"] # len(attnmat) = n_layer
            attnmat  = torch.mean(torch.cat(attnmat, axis=0), axis=1)
            attnmat  = attnmat + torch.eye(attnmat.shape[-1])
            attnmat  = attnmat / attnmat.sum(axis=-1).unsqueeze(-1)
            attnmat_residual = None
            for i in range(attnmat.shape[0]):
                if attnmat_residual is None:
                    attnmat_residual = attnmat[i]
                else:
                    attnmat_residual = torch.matmul(attnmat[i], attnmat_residual)
            mask   = attnmat_residual[0, 1:].reshape(-1, int(np.sqrt(attnmat_residual.shape[-1]))).detach().numpy()
            mask   = (cv2.resize(mask / mask.max(), img.size) * 255).astype(np.uint8)
            mask   = np.concatenate([mask.reshape(*mask.shape, 1) for _ in range(3)], axis=-1)
            imgwk  = np.concatenate([pil2cv(img), mask], axis=1)
            if resultwk is None:
                resultwk = imgwk
            else:
                resultwk = np.concatenate([resultwk, imgwk], axis=0)
        result.append(resultwk.copy())
    result = np.concatenate(result, axis=1)
    result = cv2.resize(result, (1024, 1024))
    cv2.imshow("test", result)
    cv2.waitKey(0)
    cv2.imwrite("imgattnmap.png", result)