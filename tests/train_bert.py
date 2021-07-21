import torch, kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import NewsPaperDataLoader
from kktorch.nn.configmod import ConfigModule


if __name__ == "__main__":
    # config file
    fjson = "../kktorch/model_zoo/huggingface/bert.json"

    # load config file and create network
    network = ConfigModule(
        fjson,
        ## You can override the config settings.
        user_parameters={
            "___n_node": 768,
            "___dict_freeze": {"BertEncoder": 8},
            "___n_classes": 4
        },
    )
    """
    >>> [dataloader_train.dataset[0][0], dataloader_train.dataset[1][0]]
    ['Astronauts will be soon Sipping Coffee at ISS', 'As the World Wide Web turns 25, take a look back at its beginning']
    >>> network.tokenizer.tokenize([dataloader_train.dataset[0][0], dataloader_train.dataset[1][0]])
    ['astronauts', 'will', 'be', 'soon', 'sipping', 'coffee', 'at', 'iss', 'as', 'the', 'world', 'wide', 'web', 'turns', '25', ',', 'take', 'a', 'look', 'back', 'at', 'its', 'beginning']
    >>> network.tokenizer([dataloader_train.dataset[0][0], dataloader_train.dataset[1][0]])
    {'input_ids': [[101, 25881, 2097, 2022, 2574, 24747, 4157, 2012, 26354, 102], [101, 2004, 1996, 2088, 2898, 4773, 4332, 2423, 1010, 2202, 1037, 2298, 2067, 2012, 2049, 2927, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
    >>> network.tokenizer([dataloader_train.dataset[0][0], dataloader_train.dataset[1][0]], **{"padding": True, "max_length": 512, "truncation": True})
    {'input_ids': [[101, 25881, 2097, 2022, 2574, 24747, 4157, 2012, 26354, 102, 0, 0, 0, 0, 0, 0, 0], [101, 2004, 1996, 2088, 2898, 4773, 4332, 2423, 1010, 2202, 1037, 2298, 2067, 2012, 2049, 2927, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
    """

    # dataloader
    dataloader_train = NewsPaperDataLoader(
        network.tokenizer, tokenizer_params_input={"padding": True, "max_length": network.tokenizer.model_max_length, "truncation": True}, aftprocs=[lambda x, y: [dict(x), y]],
        root='./data', train=True,  download=True, batch_size=64, shuffle=True,  num_workers=4
    )
    dataloader_valid = NewsPaperDataLoader(
        network.tokenizer, tokenizer_params_input={"padding": True, "max_length": network.tokenizer.model_max_length, "truncation": True}, aftprocs=[lambda x, y: [dict(x), y]],
        root='./data', train=False, download=True, batch_size=64, shuffle=False, num_workers=4
    )

    # trainer
    trainer = Trainer(
        network,
        losses_train=torch.nn.CrossEntropyLoss(),
        losses_valid=[[torch.nn.CrossEntropyLoss(), kktorch.nn.Accuracy()]],
        losses_train_name="ce",
        losses_valid_name=[["ce", "acc"]],
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=0.0001)}, 
        dataloader_train =dataloader_train,
        dataloader_valids=dataloader_valid,
        epoch=1000, valid_step=10, print_step=100, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()