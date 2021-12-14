import copy
import numpy as np
from PIL import Image 
import torch, kktorch
from torchvision import transforms
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import PASCALvoc2012DataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import DINOLoss
from kktorch.util.image.transforms import ResizeFixRatio


class MyTrainer(Trainer):
    def process_data_train_pre(self, input):
        return [input, ] if isinstance(input, torch.Tensor) else input
    def process_data_valid_pre(self, input):
        return [input, ] if isinstance(input, torch.Tensor) else input
    def process_data_train_aft(self, input):
        return [input[0], ]
    def process_data_valid_aft(self, input):
        return [input[0], ]
    def process_label_aft(self, label, input):
        return [input[1], ]
    def aftproc_update_weight(self):
        self.network.update_teacher_weight()


class MyPredictor(Trainer):
    def process_data_train_pre(self, input):
        return [input, ] if isinstance(input, torch.Tensor) else input
    def process_data_valid_pre(self, input):
        return [input, ] if isinstance(input, torch.Tensor) else input
    def process_data_train_aft(self, input):
        return [input[0], ]
    def process_data_valid_aft(self, input):
        return [input[0], ]


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


def aug_train(size: int, scale=(0.4, 1.0), p_blue: float=1.0, p_sol: float=0.0):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=scale, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=p_blue),
        transforms.RandomSolarize(128.0, p=p_sol),
        transforms.ToTensor(),
        transforms.Normalize(
            PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
            PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
        ),
    ])
def aug_valid(size: int, scale=(0.4, 1.0)):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=scale, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
            PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
        ),
    ])


if __name__ == "__main__":
    # load config file and create network
    network = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/dino/dino_vit.json", 
        user_parameters={
            "___n_layer": 4,
            "___n_dim": 256,
            "___n_head": 8,
            "___in_channels": 3,
            "___dropout_p": 0.0,
            "___patch_size": 16,
            "___img_size": 224,
            "___n_projection": 64
        },
    )
    network = TeacherStudent(network, update_rate=0.996)

    # dataloader. multi crop 2 x 224x224, 4 x 96x96
    dataloader_train = PASCALvoc2012DataLoader(
        train=True, download=True, batch_size=32, shuffle=True, drop_last=True,
        transforms=[
            aug_train(224, scale=(0.4, 1.0),  p_blue=1.0, p_sol=0.0),
            aug_train(224, scale=(0.4, 1.0),  p_blue=0.5, p_sol=0.1),
            aug_train(224, scale=(0.4, 1.0),  p_blue=0.1, p_sol=0.2),
            aug_train( 96, scale=(0.05, 0.4), p_blue=0.5, p_sol=0.0),
            aug_train( 96, scale=(0.05, 0.4), p_blue=0.5, p_sol=0.0),
            aug_train( 96, scale=(0.05, 0.4), p_blue=0.5, p_sol=0.0),
            aug_train( 96, scale=(0.05, 0.4), p_blue=0.5, p_sol=0.0),
            aug_train( 96, scale=(0.05, 0.4), p_blue=0.5, p_sol=0.0),
            aug_train( 96, scale=(0.05, 0.4), p_blue=0.5, p_sol=0.0),
        ],
        num_workers=32
    )

    # trainer
    trainer = MyTrainer(
        network,
        losses_train=DINOLoss(temperature_s=0.1, temperature_t=0.04, update_rate=0.9, n_global_view=3),
        losses_train_name="dino",
        optimizer={
            "optimizer": torch.optim.AdamW, 
            "params": dict(lr=1e-3)
        }, 
        dataloader_train=dataloader_train,
        epoch=100, print_step=100, auto_mixed_precision=True, accumulation_step=1, clip_grad=3.0
    )
    #trainer.reset_parameters("norm")

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # evaluation setup. multi crop 2 x 224x224
    dataloader_train = PASCALvoc2012DataLoader(
        train=True, download=True, batch_size=48, shuffle=False, drop_last=False, num_workers=8,
        transforms=[
            aug_train(224, scale=(0.4, 1.0)),
            aug_train(224, scale=(0.4, 1.0)),
            aug_train(224, scale=(0.4, 1.0)),
            aug_train(224, scale=(0.4, 1.0)),
        ],
    )
    dataloader_valid = PASCALvoc2012DataLoader(
        train=False, download=True, batch_size=64, shuffle=False, drop_last=False, num_workers=8,
        transforms=[
            aug_train(224, scale=(0.4, 1.0)),
            aug_train(224, scale=(0.4, 1.0)),
            aug_train(224, scale=(0.4, 1.0)),
            aug_train(224, scale=(0.4, 1.0)),
        ]
    )

    #trainer.load("/path/to/model/weight.pth")
    predictor = MyPredictor(network, auto_mixed_precision=True)
    predictor.load("save_output_dino_effi_1/model_7900.pth")
    predictor.to_cuda()

    # predict
    x_train, y_train = predictor.predict(dataloader_train, is_label=True, sample_size=0)
    x_valid, y_valid = predictor.predict(dataloader_valid, is_label=True, sample_size=0)
    x_train = x_train.mean(axis=1) # mean multi crop output
    x_valid = x_valid.mean(axis=1) # mean multi crop output

    # knn
    import faiss
    index = faiss.IndexFlatL2(x_train.shape[-1])
    index.add(x_train.astype(np.float32))
    D, I = index.search(x_valid.astype(np.float32), k=5)
    """
    >>> D
    array([[0.00050521, 0.00052023, 0.00115323, 0.00134861, 0.00158119],
        [0.00081217, 0.00113809, 0.00161076, 0.00280142, 0.00340712],
        [0.00166631, 0.00363696, 0.00414395, 0.00453103, 0.00543332],
        ...,
        [0.0020014 , 0.00202489, 0.00245035, 0.00355589, 0.00363195],
        [0.00039721, 0.00060582, 0.00060976, 0.0007081 , 0.00074315],
        [0.00186503, 0.0020678 , 0.00261247, 0.00307369, 0.00347137]],
        dtype=float32)
    >>> I
    array([[ 707, 1862, 3145, 3080, 3209],
        [2751, 1747, 1371, 4199, 5249],
        [  96,  495, 3470, 4979, 2392],
        ...,
        [3671, 1526, 1756, 5363, 4205],
        [2506, 5055, 4014, 4424, 1412],
        [5066, 3910, 3026, 4915, 5102]])
    """
    n = (y_train[I, :].mean(axis=1).argmax(axis=1) == y_valid.argmax(axis=1)).sum()
    print(len(dataloader_valid), n, f"acc: {n / len(dataloader_valid)}")

    # plot
    import cv2
    base_size = 200
    for i in range(1, 20):
        img = ResizeFixRatio(base_size, "height")(dataloader_valid.dataset.get_image(i))
        for j in I[i]: img = np.concatenate([img, ResizeFixRatio(base_size, "height")(dataloader_train.dataset.get_image(j))], axis=1) 
        cv2.imwrite(f"knn{i}.png", img)
