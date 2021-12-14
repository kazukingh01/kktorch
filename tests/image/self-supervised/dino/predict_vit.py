import copy
import numpy as np
from PIL import Image 
import torch, kktorch

import kktorch.util.image as tfms
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import PASCALvoc2012DataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.util.image.transforms import ResizeFixRatio
from kktorch.util.image import pil2cv


class MyTrainer(Trainer):
    def process_data_train_pre(self, input):
        return [input, ] if isinstance(input, torch.Tensor) else input
    def process_data_valid_pre(self, input):
        return [input, ] if isinstance(input, torch.Tensor) else input


class TeacherStudent(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, update_rate: float=0.9):
        super().__init__()
        self.student = copy.deepcopy(model)
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters(): p.requires_grad = False
        self.update_rate = update_rate
    def forward(self, input):
        output_s = self.student(input)
        with torch.no_grad():
            output_t = self.teacher(input)
        return output_s, output_t
    def parameters(self, recurse: bool = True):
        return self.student.parameters(recurse=recurse)
    def update_teacher_weight(self):
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data.mul_(self.update_rate).add_((1 - self.update_rate) * param_s.detach().data)


def aug_valid(size: int, scale=(0.4, 1.0)):
    return tfms.Compose([
        tfms.RandomResizedCrop(size, scale=scale, interpolation=Image.BICUBIC),
        tfms.ToTensor(),
        tfms.Normalize(
            PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
            PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
        ),
    ])


if __name__ == "__main__":
    # parser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight")
    args = parser.parse_args()

    # load config file and create network
    img_size = 224
    network  = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/dino/dino_vit.json", 
        user_parameters={
            "___n_layer": 6,
            "___n_dim": 256,
            "___n_head": 8,
            "___in_channels": 3,
            "___dropout_p": 0.0,
            "___patch_size": 16,
            "___img_size": img_size,
            "___n_projection": 128
        },
    )
    network = TeacherStudent(network, update_rate=0.98)

    # trainer
    predictor = MyTrainer(network)
    predictor.load(args.weight)
    predictor.to_cuda()

    # dataloader. evaluation setup. multi crop 2 x 224x224
    dataloader_train = PASCALvoc2012DataLoader(
        train=True, download=True, batch_size=32, shuffle=False, drop_last=False, num_workers=8,
        transforms=[
            aug_valid(224, scale=(0.4, 1.0)),
            aug_valid(224, scale=(0.4, 1.0)),
            aug_valid(224, scale=(0.4, 1.0)),
            aug_valid(224, scale=(0.4, 1.0)),
        ], is_label_binary_class=True
    )
    dataloader_valid = PASCALvoc2012DataLoader(
        train=False, download=True, batch_size=32, shuffle=False, drop_last=False, num_workers=8,
        transforms=[
            aug_valid(224, scale=(0.4, 1.0)),
            aug_valid(224, scale=(0.4, 1.0)),
            aug_valid(224, scale=(0.4, 1.0)),
            aug_valid(224, scale=(0.4, 1.0)),
        ], is_label_binary_class=True
    )

    # predict
    x_train, y_train = predictor.predict(dataloader_train, is_label=True, sample_size=0)
    x_valid, y_valid = predictor.predict(dataloader_valid, is_label=True, sample_size=0)
    x_train = x_train[0].mean(axis=1) # mean multi crop output
    x_valid = x_valid[0].mean(axis=1) # mean multi crop output

    # knn prediction
    import faiss
    index = faiss.IndexFlatL2(x_train.shape[-1])
    index.add(x_train.astype(np.float32))
    D, I = index.search(x_valid.astype(np.float32), k=10)
    """
    >>> D
    array([[0.00050521, 0.00052023, 0.00115323, 0.00134861, 0.00158119],
        [0.00081217, 0.00113809, 0.00161076, 0.00280142, 0.00340712],
        ...,
        [0.00039721, 0.00060582, 0.00060976, 0.0007081 , 0.00074315],
        [0.00186503, 0.0020678 , 0.00261247, 0.00307369, 0.00347137]],
        dtype=float32)
    >>> I
    array([[ 707, 1862, 3145, 3080, 3209],
        [2751, 1747, 1371, 4199, 5249],
        ...,
        [2506, 5055, 4014, 4424, 1412],
        [5066, 3910, 3026, 4915, 5102]])
    """
    n = (y_train[I, :].mean(axis=1).argmax(axis=1) == y_valid.argmax(axis=1)).sum()
    print(len(dataloader_valid), n, f"acc: {n / len(dataloader_valid)}")

    # knn plot
    import cv2
    imgs, base_size = [], 200
    for i in range(1, 15):
        img = ResizeFixRatio(base_size, "height")(dataloader_valid.dataset.get_image(i))
        for j in I[i]: img = np.concatenate([img, ResizeFixRatio(base_size, "height")(dataloader_train.dataset.get_image(j))], axis=1) 
        imgs.append(img)
    height_max = max([x.shape[1] for x in imgs])
    imgs = [ResizeFixRatio(height_max, "width")(x) for x in imgs]
    imgs = np.concatenate(imgs, axis=0)
    imgs = cv2.resize(imgs, (imgs.shape[0]//5, imgs.shape[1]//5))
    cv2.imwrite(f"knn_vit.png", imgs)

    # attention map
    result = []
    for I in range(4):
        resultwk = None
        for J in range(8):
            img     = dataloader_valid.dataset.get_image(I*8+J, "pil")
            img     = tfms.RandomResizedCrop(224, scale=(0.99, 1.0))(img)
            input   = tfms.Compose([
                tfms.ToTensor(),
                tfms.Normalize(
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
                    PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
                )
            ])(img)
            input   = input.unsqueeze(0).unsqueeze(0)
            output  = network.student.forward_debug([input.to("cuda")])
            attnmat = [z[1].to("cpu") for x in output for y, z in x.items() if y == "selfattention"] # len(attnmat) = n_layer
            attnmat = torch.mean(torch.cat(attnmat, axis=0), axis=1)
            attnmat = attnmat + torch.eye(attnmat.shape[-1])
            attnmat = attnmat / attnmat.sum(axis=-1).unsqueeze(-1)
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
    result = cv2.resize(result, (512, 512))
    cv2.imshow("test", result)
    cv2.waitKey(0)
    cv2.imwrite("imgattnmap_vit.png", result)
