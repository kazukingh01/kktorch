import torch, kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import LivedoorNewsDataLoader
from kktorch.nn.configmod import ConfigModule


if __name__ == "__main__":
    # config file
    fjson = "../kktorch/model_zoo/huggingface/bert_jp_seq2seq.json"

    # load config file and create network
    network = ConfigModule(
        fjson,
        ## You can override the config settings.
        user_parameters={
            "___n_node": 768,
            "___dict_freeze": {"BertEncoder": 8},
        },
    )

    # dataloader
    dataloader_train = LivedoorNewsDataLoader(
        network.tokenizer, 
        tokenizer_params_input={"padding": True, "max_length": 512, "truncation": True},
        tokenizer_params_label={"padding": "max_length", "max_length": 512, "truncation": True},
        aftprocs_input=[lambda x: dict(x)], aftprocs_label=[lambda x: dict(x)["input_ids"]],
        root='./data', train=True,  download=True, columns=["body", "title"], batch_size=2, shuffle=True,  num_workers=4
    )
    dataloader_valid = LivedoorNewsDataLoader(
        network.tokenizer, 
        tokenizer_params_input={"padding": True, "max_length": 512, "truncation": True},
        tokenizer_params_label={"padding": "max_length", "max_length": 512, "truncation": True},
        aftprocs_input=[lambda x: dict(x)], aftprocs_label=[lambda x: dict(x)["input_ids"]],
        root='./data', train=False, download=True, columns=["body", "title"], batch_size=64, shuffle=False, num_workers=4
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