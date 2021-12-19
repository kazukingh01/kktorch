import numpy as np
import torch
import torchvision.transforms as tfms
from PIL import Image

import kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MNISTDataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import DeepSVDDLoss


def create_circle(C: np.ndarray, R: float, niter: int=100):
    n_dim    = C.shape[0]
    rand_dim = np.stack([np.random.permutation(np.arange(n_dim)) for _ in range(niter)])
    rand_pm  = np.random.randint(0, 2, (niter, n_dim))
    rand_pm[rand_pm == 0] = -1
    ndf      = []
    for i in range(niter):
        rand_vec = np.zeros(n_dim)
        _R = R
        for j in rand_dim[i][:-1]:
            point = np.random.rand(1)[0] * _R * rand_pm[i, j]
            rand_vec[j] = point
            _R = np.sqrt(_R ** 2 - point ** 2)
        j = rand_dim[i][-1]
        point = np.sqrt(R ** 2 - (rand_vec ** 2).sum())
        rand_vec[j] = point * rand_pm[i, j]
        ndf.append(rand_vec)
    return np.stack(ndf) + C


class MyTrainer(Trainer):
    def process_label_pre(self, label, input):
        # return label as input for image reconstruction
        return input


if __name__ == "__main__":
    str_result = []
    for i in range(0, 10):
        # network
        dim_z   = 16
        encoder = ConfigModule(
            f"/{kktorch.__path__[0]}/model_zoo/autoencoder/encoder.json", 
            user_parameters={
                "___in_channels": 1,
                "___init_dim": 8,
                "___n_layers": 4,
                "___z": dim_z,
            }
        )
        decoder = ConfigModule(
            f"/{kktorch.__path__[0]}/model_zoo/autoencoder/decoder.json", 
            user_parameters={
                "___in_channels": 1,
                "___init_size": 2,
                "___alpha": 32,
                "___n_layers": 4,
                "___z": dim_z,
            }
        )
        # dataloader
        dataloader_train = MNISTDataLoader(
            train=True, download=True, batch_size=256, shuffle=False,
            transforms=[tfms.Resize(32, interpolation=Image.BICUBIC),tfms.ToTensor()], 
            classes_targets=[j for j in range(10) if j != i]
        )
        # Pre-training ( autoencoder ) 
        trainer = MyTrainer(
            torch.nn.Sequential(encoder, decoder),
            losses_train=torch.nn.MSELoss(), losses_train_name=["mse"],
            optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-4)},
            dataloader_train =dataloader_train,
            epoch=1, valid_step=10, print_step=500, 
        )
        # to cuda
        trainer.to_cuda()
        # training
        trainer.train()

        # Main-training ( DeepSVDD ) 
        loss    = DeepSVDDLoss(nu=0.05, n_update_C=(len(dataloader_train) // dataloader_train.batch_size))
        trainer = Trainer(
            encoder,
            losses_train=loss, losses_train_name=["svdd"],
            optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-4, weight_decay=1e-1)},
            dataloader_train =dataloader_train,
            epoch=5, valid_step=10, print_step=500, 
        )
        # to cuda
        trainer.to_cuda()
        # training
        trainer.train()
        # scatter plot
        import matplotlib.pyplot as plt
        plt.figure(figsize = (6, 4))
        output_train, label_train = trainer.predict(dataloader=dataloader_train, is_label=True, sample_size=-1)
        dataloader_train = MNISTDataLoader(
            train=True, download=True, batch_size=256, shuffle=False,
            transforms=[tfms.ToTensor()], classes_targets=[i,]
        )
        output_test, label_test = trainer.predict(dataloader=dataloader_train, is_label=True, sample_size=-1)
        R = loss.R.to("cpu").item()
        C = loss.C.to("cpu").detach().numpy()
        ## accuracy
        ndf_bool = np.sqrt(((output_test - C) ** 2).sum(axis=-1)) > R
        result   = f"total: {ndf_bool.shape[0]}, acc: {round(ndf_bool.sum() / ndf_bool.shape[0], 2)}"
        ## t-SNE
        from MulticoreTSNE import MulticoreTSNE as TSNE
        tsne   = TSNE(n_jobs=32)
        output = tsne.fit_transform(np.concatenate([output_train, output_test], axis=0))
        n_train, n_test = output_train.shape[0], output_test.shape[0]
        for n_min, n_max, color, label in zip(np.cumsum([0, n_train]), np.cumsum([n_train, n_test]), ["b","r"], ["normal","anomaly"]):
            plt.scatter(output[n_min:n_max, 0], output[n_min:n_max, 1], label=label, color=color, s=5, alpha=0.05)
        str_result.append(result)
        plt.title(f"anomaly: {i}, {result}")
        plt.legend()
        plt.savefig(f"mnist_tsne_{i}_anomaly.png")
        print(result)
    print(str_result)



