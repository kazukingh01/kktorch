import torch, kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import NewsPaperDataLoader
from kktorch.nn.configmod import ConfigModule


if __name__ == "__main__":
    # config file
    fjson = "../kktorch/model_zoo/huggingface/huggingface.json"

    # load config file and create network
    network = ConfigModule(fjson)

    # dataloader
    dataloader_train = NewsPaperDataLoader(root='./data', train=True,  download=True, batch_size=64, shuffle=True,  num_workers=4)
    dataloader_test  = NewsPaperDataLoader(root='./data', train=False, download=True, batch_size=64, shuffle=False, num_workers=4)
    dataloader_train.dataset.set_tokenizer(network.tokenizer)
    dataloader_test. dataset.set_tokenizer(network.tokenizer)
    raise

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