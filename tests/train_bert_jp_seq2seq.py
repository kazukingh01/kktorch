import torch, kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import LivedoorNewsDataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.util.text.transforms import shift_right_decoder_input, decoder_attention_mask, generate

from transformers import T5ForConditionalGeneration, T5Tokenizer, EncoderDecoderModel

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
    
    # dataloader
    dataloader_train = LivedoorNewsDataLoader(
        network.tokenizer, 
        tokenizer_params_input={"padding": "longest", "max_length": 1024, "truncation": True},
        tokenizer_params_label={"padding": "longest", "max_length": 1024, "truncation": True},
        aftprocs=[
            lambda x, y: [
                {
                    "input_ids": x["input_ids"], "attention_mask": x["attention_mask"], 
                    "decoder_input_ids": shift_right_decoder_input()(
                        y["input_ids"], decoder_start_token_id=decoder_start_token_id, 
                        eos_token_id=network.tokenizer.eos_token_id, padding_token_id=decoder_start_token_id
                    ),
                    "decoder_attention_mask": decoder_attention_mask()(y["input_ids"], decoder_start_token_id),
                }, y["input_ids"]
            ],
        ], 
        root='./data', train=True,  download=True, columns=["body", "title"], batch_size=16, shuffle=True,  num_workers=0
    )
    dataloader_valid = LivedoorNewsDataLoader(
        network.tokenizer, 
        tokenizer_params_input={"padding": "longest"},
        tokenizer_params_label={"padding": "longest"},
        aftprocs=[
            lambda x, y: [
                {
                    "input_ids": x["input_ids"], "attention_mask": x["attention_mask"], 
                    "decoder_input_ids": shift_right_decoder_input()(
                        y["input_ids"], decoder_start_token_id=decoder_start_token_id, 
                        eos_token_id=network.tokenizer.eos_token_id, padding_token_id=decoder_start_token_id
                    ),
                    "decoder_attention_mask": decoder_attention_mask()(y["input_ids"], decoder_start_token_id),
                }, y["input_ids"]
            ],
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
        epoch=100, valid_step=10, print_step=100, accumulation_step=1,
    )

    # load
    trainer.load("./output_train_bert_jp_seq2seq_20210731192854/model_33347.pth")
    network.huggingface[0].freeze()
    # to cuda
    trainer.to_cuda()

    # training
    #trainer.train()

    # generate
    x, y = dataloader_valid[1]
    output = generate()(network, x, bos_token_id=0, eos_token_id=1)
    network.tokenizer.decode(output.tolist()[0])
    network.tokenizer.decode(y.tolist()[0])
