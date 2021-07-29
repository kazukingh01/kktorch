import torch, kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import LivedoorNewsDataLoader
from kktorch.nn.configmod import ConfigModule


from transformers import T5ForConditionalGeneration, T5Tokenizer

if __name__ == "__main__":
    # config file
    fjson = "../kktorch/model_zoo/huggingface/bert_jp_seq2seq.json"

    # load config file and create network
    network = ConfigModule(
        fjson,
        ## You can override the config settings.
        user_parameters={
            "___n_node": 768,
            "___freeze_layers": [
                "^shared\\.weight",
                "^encoder\\.block\\.[0-9]\\.layer",
                "^decoder\\.block\\.[0-9]\\.layer"
            ]
        },
    )

    decoder_start_token_id = network.huggingface_config.decoder_start_token_id
    def shift_right(input_ids, decoder_start_token_id: int=0):
        # see: https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L770-L794
        shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
        shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        return shifted_input_ids

    # dataloader
    dataloader_train = LivedoorNewsDataLoader(
        network.tokenizer, 
        tokenizer_params_input={"padding": "longest"},
        tokenizer_params_label={"padding": "longest"},
        aftprocs=[
            lambda x, y: [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"], "decoder_input_ids": shift_right(torch.Tensor(y["input_ids"]), decoder_start_token_id)}, y["input_ids"]],
        ], 
        root='./data', train=True,  download=True, columns=["body", "title"], batch_size=1, shuffle=True,  num_workers=0
    )
    dataloader_valid = LivedoorNewsDataLoader(
        network.tokenizer, 
        tokenizer_params_input={"padding": "longest"},
        tokenizer_params_label={"padding": "longest"},
        aftprocs=[
            lambda x, y: [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"], "decoder_input_ids": shift_right(torch.Tensor(y["input_ids"]), decoder_start_token_id)}, y["input_ids"]],
        ], 
        root='./data', train=False, download=True, columns=["body", "title"], batch_size=1, shuffle=False, num_workers=0
    )

    class MyLoss(torch.nn.CrossEntropyLoss):
        def __init__(self, embedding_dim: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.embedding_dim = embedding_dim
        def forward(self,input, target):
            return super().forward(input.reshape(-1, self.embedding_dim), target.reshape(-1))

    # trainer
    trainer = Trainer(
        network,
        losses_train=MyLoss(network.huggingface[2].param.shape[0], ignore_index=0),
        losses_valid=[[MyLoss(network.huggingface[2].param.shape[0], ignore_index=0)]],
        losses_train_name="ce",
        losses_valid_name=[["ce"]],
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-5)}, 
        dataloader_train =dataloader_train,
        dataloader_valids=dataloader_valid,
        epoch=10, valid_step=10, print_step=100, accumulation_step=10,
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()