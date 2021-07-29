import numpy as np
import torch, kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import MNISTDataLoader
from kktorch.nn.configmod import ConfigModule
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


if __name__ == "__main__":
    # config file
    fjson = "../kktorch/model_zoo/timm/timm.json"

    # load config file and create network
    network = ConfigModule(
        fjson,
        ## You can override the config settings.
        user_parameters={
            "___n_classes": 10,
            "___freeze_layers": [
                "^conv_stem\\.",
                "^bn1\\.",
                "^blocks\\.[0-5]\\.",
            ]
        },
    )

    # dataloader
    dataloader_train = MNISTDataLoader(
        root='./data', train=True, download=True, batch_size=16, shuffle=True,
        transform=[
            transforms.ToTensor(), transforms.Resize([256,256]), transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ], num_workers=4
    )
    dataloader_valid = MNISTDataLoader(
        root='./data', train=False, download=True, batch_size=16, shuffle=False,
        transform=[
            transforms.ToTensor(), transforms.Resize([256,256]), transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ], num_workers=4
    )

    # trainer
    trainer = Trainer(
        network,
        losses_train=torch.nn.CrossEntropyLoss(),
        losses_valid=[[torch.nn.CrossEntropyLoss(), kktorch.nn.Accuracy()]],
        losses_train_name="ce",
        losses_valid_name=[["ce", "acc"]],
        optimizer={"optimizer": torch.optim.SGD, "params": dict(lr=0.001, weight_decay=0)}, 
        dataloader_train =dataloader_train,
        dataloader_valids=dataloader_valid,
        epoch=1000, valid_step=10, print_step=100, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # predict
    x, y = trainer.predict(dataloader_valid, is_label=True, sample_size=1)
    """
    >>> x
    array([[-2.0741627 ,  0.68259317, -0.6578207 , -2.449573  ,  0.6852776 ,
            -0.23112217, -1.986605  ,  6.5254917 , -2.1286929 ,  0.60230875],
        [-1.0165738 , -2.6702724 ,  1.9001055 ,  0.14558092, -1.4260466 ,
            -0.1822576 ,  0.8647478 ,  1.8212023 , -0.41427714,  0.0460536 ],
        [-1.6494608 ,  7.724181  , -2.6105778 , -3.8918717 ,  2.7686055 ,
            -1.0559607 ,  0.7866792 ,  2.1898596 , -1.5055339 , -1.9467576 ],
        [ 5.527744  , -1.6580219 , -1.4025593 , -1.2036648 , -0.03430375,
            -1.2221682 ,  1.2400321 , -3.752311  ,  1.2312728 ,  0.3319189 ],
        [-1.0391644 ,  0.9002493 ,  1.5681363 , -3.1391873 ,  3.505603  ,
            0.29428154,  0.52986884,  0.7295419 , -2.2827234 , -0.8697704 ],
        [-0.19168651,  8.324283  , -2.0710742 , -4.25594   ,  0.46990806,
            -3.176092  ,  0.85418326,  4.258133  , -3.518204  ,  0.01443487],
        [-3.4022677 , -1.1207379 , -0.17137432, -0.96513057,  3.8576777 ,
            -0.01907951, -0.58751374,  0.248315  ,  0.4785603 , -0.21121238],
        [-1.5622996 , -1.6454175 ,  0.20492463, -0.79972726, -0.08901856,
            -1.3058845 ,  1.253424  , -0.49440232,  1.807915  ,  1.9161913 ],
        [-1.9300096 , -2.5711632 ,  2.085733  ,  0.691076  , -1.1643496 ,
            2.1700268 ,  1.4171002 , -1.0454737 , -0.25349274, -0.9915173 ],
        [ 0.26798213, -4.6203785 ,  0.4317138 ,  0.4473066 , -1.4402165 ,
            -1.0867584 ,  0.03723752, -0.5489694 ,  1.210747  ,  3.5362854 ],
        [ 4.4743648 , -1.6863859 ,  2.0900822 ,  0.02468932, -1.6946982 ,
            0.47598726,  0.34381023,  0.6902799 , -1.8556209 , -0.25936794],
        [ 2.005881  , -3.3796053 ,  1.6180174 , -0.7765627 , -0.7969086 ,
            -0.37286058,  3.1244335 , -2.3507085 ,  0.21931544, -0.7251687 ],
        [ 0.85915875, -3.6328726 , -2.6912234 , -2.7543392 , -0.34299797,
            -1.6961904 , -0.18794918,  1.2001436 , -0.6186386 ,  7.2392936 ],
        [ 6.240278  , -1.0156105 ,  0.1842934 , -0.86775285, -1.616312  ,
            -2.2262402 ,  1.8089914 , -1.6126794 ,  0.21113758,  1.5271149 ],
        [-1.5657198 ,  3.4548442 , -1.0834526 , -1.2986451 ,  2.1768644 ,
            -2.4513884 , -0.7921509 , -0.2812668 ,  0.8533096 , -2.100736  ],
        [-0.32296157, -2.5020897 ,  1.2819008 ,  2.3290043 , -0.76090467,
            3.5417843 , -0.43499014,  0.8156693 , -0.34818822, -2.0067875 ]],
        dtype=float32)
    >>> y
    array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5])
    >>> (np.argmax(x, axis=1) == y).sum() / y.shape[0]
    1.0
    """
