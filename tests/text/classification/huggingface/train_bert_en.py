import torch, kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import NewsPaperDataLoader
from kktorch.nn.configmod import ConfigModule


if __name__ == "__main__":
    # load config file and create network
    network = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/huggingface/bert_en_classify.json", 
        user_parameters={
            "___n_node": 768,
            "___freeze_layers": [
                "^embeddings",
                "^encoder\\.layer\\.[0-9]\\.",
            ],
            "___n_classes": 4
        },
    )
    """
    >>> dataloader_train.dataset[0]
    ('iPhone 6 Release Date Pushed Back Due to Issues With Battery', 1)
    >>> dataloader_train.dataset[1]
    ('Samsung Galaxy S4 vs Galaxy S3: Budget-Friendly Legends', 1)
    >>> [dataloader_train.dataset[0][0], dataloader_train.dataset[1][0]]
    ['iPhone 6 Release Date Pushed Back Due to Issues With Battery', 'Samsung Galaxy S4 vs Galaxy S3: Budget-Friendly Legends']
    >>> network.tokenizer.tokenize([dataloader_train.dataset[0][0], dataloader_train.dataset[1][0]])
    ['iphone', '6', 'release', 'date', 'pushed', 'back', 'due', 'to', 'issues', 'with', 'battery', 'samsung', 'galaxy', 's', '##4', 'vs', 'galaxy', 's', '##3', ':', 'budget', '-', 'friendly', 'legends']
    >>> network.tokenizer([dataloader_train.dataset[0][0], dataloader_train.dataset[1][0]])
    {'input_ids': [[101, 18059, 1020, 2713, 3058, 3724, 2067, 2349, 2000, 3314, 2007, 6046, 102], [101, 19102, 9088, 1055, 2549, 5443, 9088, 1055, 2509, 1024, 5166, 1011, 5379, 9489, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
    >>> network.tokenizer([dataloader_train.dataset[0][0], dataloader_train.dataset[1][0]], **{"padding": True, "max_length": 512, "truncation": True})
    {'input_ids': [[101, 18059, 1020, 2713, 3058, 3724, 2067, 2349, 2000, 3314, 2007, 6046, 102, 0, 0], [101, 19102, 9088, 1055, 2549, 5443, 9088, 1055, 2509, 1024, 5166, 1011, 5379, 9489, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
    """

    # dataloader
    dataloader_train = NewsPaperDataLoader(
        network.tokenizer, tokenizer_params_input={"padding": True, "max_length": network.tokenizer.model_max_length, "truncation": True}, aftprocs=[lambda x, y: [dict(x), y]],
        train=True,  download=True, batch_size=128, shuffle=True,  num_workers=8
    )
    dataloader_valid = NewsPaperDataLoader(
        network.tokenizer, tokenizer_params_input={"padding": True, "max_length": network.tokenizer.model_max_length, "truncation": True}, aftprocs=[lambda x, y: [dict(x), y]],
        train=False, download=True, batch_size=128, shuffle=False, num_workers=8
    )
    """
    >>> dataloader_train.sample()
    [{'input_ids': tensor([[ 101, 2064, 2017,  ...,    0,    0,    0],
        [ 101, 5117, 5206,  ...,    0,    0,    0],
        [ 101, 8201, 9304,  ...,    0,    0,    0],
        ...,
        [ 101, 9543, 4063,  ...,    0,    0,    0],
        [ 101, 4907, 1005,  ...,    0,    0,    0],
        [ 101, 1048, 1005,  ...,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]])}, tensor([1, 2, 2, 2, 0, 1, 2, 0, 2, 1, 3, 3, 2, 2, 2, 0, 1, 2, 2, 2, 0, 2, 1, 1,
        1, 2, 1, 0, 2, 1, 0, 2, 1, 0, 1, 0, 1, 2, 1, 1, 3, 2, 0, 0, 2, 1, 3, 0,
        0, 2, 2, 0, 2, 0, 0, 2, 2, 0, 1, 0, 1, 2, 2, 2])]
    """

    # trainer
    trainer = Trainer(
        network,
        losses_train=torch.nn.CrossEntropyLoss(),
        losses_valid=[[torch.nn.CrossEntropyLoss(), kktorch.nn.Accuracy()]],
        losses_train_name="ce",
        losses_valid_name=["ce", "acc"], 
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-4)}, 
        dataloader_train =dataloader_train, dataloader_valids=dataloader_valid,
        auto_mixed_precision=True, epoch=2, valid_step=200, valid_iter=10, print_step=200, 
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # predict
    x, y = trainer.predict(dataloader_valid, is_label=True, sample_size=1)
    """
    >>> x
    array([[-1.0524765 , -0.61803895,  1.0787393 , -1.0102578 ],
        [ 0.13710295,  0.30140278,  0.9317543 , -0.9869781 ],
        [ 1.9124478 ,  0.87749845, -0.4696013 , -0.7225257 ],
        [-0.7787878 ,  0.0382604 ,  0.85921466, -1.0726813 ],
        [ 0.8177269 ,  0.93726975, -0.3244017 , -0.13197201],
        ...
    >>> y
    array([2, 0, 0, 2, 3, 1, 2, 2, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2, 0, 1, 3, 2,
        2, 2, 2, 1, 0, 2, 0, 0, 1, 3, 0, 2, 3, 0, 2, 0, 0, 1, 2, 2, 1, 1,
        3, 0, 0, 3, 1, 2, 2, 1, 3, 1, 2, 3, 1, 2, 2, 1, 3, 2, 0, 0])
    """