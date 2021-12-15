import torch, kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import LivedoorNewsDataLoader
from kktorch.nn.configmod import ConfigModule
from kktorch.nn.loss import CrossEntropyAcrossLoss
from kktorch.util.text.transforms import shift_right_decoder_input, decoder_attention_mask, generate


if __name__ == "__main__":
    # load config file and create network
    network = ConfigModule(
        f"/{kktorch.__path__[0]}/model_zoo/huggingface/t5_jp_seq2seq.json", 
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
        train=True, download=True, columns=["body", "title"], batch_size=16, shuffle=True,  num_workers=16
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
        train=False, download=True, columns=["body", "title"], batch_size=8, shuffle=False, num_workers=16
    )

    # trainer
    trainer = Trainer(
        network,
        losses_train=CrossEntropyAcrossLoss(network.huggingface[2].param.shape[0], ignore_index=0),
        losses_valid=CrossEntropyAcrossLoss(network.huggingface[2].param.shape[0], ignore_index=0),
        losses_train_name="ce", losses_valid_name="ce",
        optimizer={"optimizer": torch.optim.AdamW, "params": dict(lr=1e-4)}, 
        dataloader_train =dataloader_train, dataloader_valids=dataloader_valid,
        epoch=5, valid_step=20, valid_iter=10, print_step=100, accumulation_step=1, auto_mixed_precision=False
    )

    # to cuda
    trainer.to_cuda()

    # training
    trainer.train()

    # generate
    x, y = dataloader_valid[0]
    """
    >>> x
    {'input_ids': tensor([[10091,    12,    39,    18,    63,    26,     3,  7337,  7948,    15,
            5569, 17496,     3,  6318, 21969,   375,  2414,  8797,  5082,    19,
            21969,   375,  2414,     5,    37,     5,  9044,   144,   603,     5,
            193,  2328,  6522, 26729,  8045,     8,  1080,    15,  4257,   730,
                4,     5, 21969,  8797,  5082,     7,     3, 15522,    20,    26,
            1592,    39,    18,    45,    26,  1613,  5611,    19,  9417,    53,
            388,     4,    17,  4438,  2446,     8, 17363, 16474,  7018,   730,
                4,  6866,     3,   159,   817,     6,  3693,  7574,    40,     3,
            15636,  3573,     3, 21969,   728,    14, 10485,   330,   781,    69,
            18345,  8690,  9213,  6481,     4,     5, 21969, 12914, 27119,    24,
            2470,    44,  2123,   790,     8,   929, 13637,     3, 11407,  1562,
                51,   151,  3534,  3040, 18345,  1972,     3,    68,    18,   285,
            5651,     6, 15446,  2682,    20,  4150,  3534,   577,     4,  3693,
                6,  4974,    19,   375,  2414,  2135, 25896,   353,  4634,    28,
                21,    26, 29729,   253,   213,    22,  9304, 23217,  2195,   485,
                3, 16455,  8444, 21969,   728, 11869,  4696,  7654,   317,     4,
                5, 21059, 23950,     3,  1080,     6, 14187, 14643,     3,   781,
            3194,  1929,  9142, 20884,    13,    87, 16215,   321, 23572,    17,
            22023,     8, 25324,    55,  3159,    87,   794,    47, 19634,    14,
                28, 13776, 25693, 20461,   781,   950,   317,  1430,    17,    42,
                3,    60,  1179,    13, 20486,  2975,    55, 18345,    28, 11350,
                4, 12409,  1330,   517,  6939,     5,     9,     5, 21969,   375,
            2414,     5,    37,     5,  9044,   144,   603,     5,   193,  2328,
            6522, 26729,   204,     2,  9417,    53,   388,     4,     2,     5,
                1]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1]]), 'decoder_input_ids': tensor([[    0,     5,  1702,  9142,   794,    24,    44,  2123,   790,     3,
            21969,   375,  2414,  8797,  5082,     8, 15636,  4257,    13]]), 'decoder_attention_mask': tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])}
    >>> y
    tensor([[    5,  1702,  9142,   794,    24,    44,  2123,   790,     3, 21969,
            375,  2414,  8797,  5082,     8, 15636,  4257,    13,     1]])
    """
    for i in range(10):
        x, y = dataloader_valid[i]
        output = generate()(network, x, bos_token_id=0, eos_token_id=1)
        print("Pred:", network.tokenizer.decode(output.tolist()[0]))
        print("GT:  ", network.tokenizer.decode(y.tolist()[0]))
