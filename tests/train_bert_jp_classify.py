import torch, kktorch
from kktorch.trainer.base import Trainer
from kktorch.data.dataloader import LivedoorNewsDataLoader
from kktorch.nn.configmod import ConfigModule


if __name__ == "__main__":
    # config file
    fjson = "../kktorch/model_zoo/huggingface/bert_jp_classify.json"

    # load config file and create network
    network = ConfigModule(
        fjson,
        ## You can override the config settings.
        user_parameters={
            "___n_node": 768,
            "___dict_freeze": {"BertEncoder": 8},
            "___n_classes": 9
        },
    )
    """
    >>> dataloader_train.dataset[0][0]
    'ビートたけしの発言に賛否両論\n2月24日配信の「メルマガNEWSポストセブン」に掲載された記事「ビートたけしの今週のオピニオン」で、ビートたけしが「大卒就職率の低下はバカ大学が増えたから」など、独自の“大学論”を展開し、ネット掲示板で反響を呼んでいる。\n\nたけしは同メルマガで大卒学生の就職率の低下に触れながら、「そもそも大学に行く必要のあるヤツラがどれだけいるのかってことなんじゃないか」「よくよく考えりゃ『猫も杓子も大学に行くようになったから、結果的に就職できないヤツが増えてるだけ』だろうよ」など、“大学全入社会”に懐疑的な見方を示した。\n\nさらに「大学になんか入らなくたって、『一流』と呼ばれる仕事人はたくさんいる。大工になったっていいし、コックや寿司職人になったっていい」と、若者の大学進学以外の選択肢を提示した。\n\nこのメルマガに対して、ネット掲示板では「正論」「大卒バカが増えたのは同意」「よく言った」などたけしの意見に肯定的な反応が目立ったが、その一方で「実際には四大卒が応募最低条件の会社が多い」「高卒では就職がもっと厳しい」「一財産成して世界的にも成功が認められてる人が言うから許される」など、たけしの意見は「大卒が有利である」という就職活動の現状からは程遠い“理想論”だという批判も見られた。\n\n\n【関連記事】\n・ビートたけし「大卒就職率の低下はバカ大学が増えたから」\n\n【関連情報】\n・【話題】 ビートたけし 「大卒就職率の低下はバカ大学が増えたから」\n'
    >>> network.tokenizer.tokenize(dataloader_train.dataset[0][0])
    ['ビート', '##た', '##けし', 'の', '発言', 'に', '賛否', '両', '##論', '2', '月', '24', '日', '配信', 'の', '「', 'メル', '##マ', '##ガ', 'NEWS', 'ポスト', 'セブン', '」', 'に', '掲載', 'さ', 'れ', 'た', '記事', '「', 'ビート', '##た', '##けし', 'の', '今', '##週', 'の', 'オ', '##ピ', '##ニオン', '」', 'で', '、', 'ビート', '##た', '##けし', 'が', '「', '大', '##卒', '就職', '率', 'の', '低下', 'は', 'バカ', '大学', 'が', '増え', 'た', 'から', '」', 'など', '、', '独自', 'の', '“', '大学', '論', '”', 'を', '展開', 'し', '、', 'ネット', '掲示板', 'で', '反響', 'を', '呼ん', 'で', 'いる', '。', 'たけし', 'は', '同', 'メル', '##マ', '##ガ', 'で', '大', '##卒', '学生', 'の', '就職', '率', 'の', '低下', 'に', '触れ', 'ながら', '、', '「', 'そもそも', '大学', 'に', '行く', '必要', 'の', 'ある', 'ヤ', '##ツ', '##ラ', 'が', 'どれ', 'だけ', 'いる', 'の', 'か', 'って', 'こと', 'な', 'ん', 'じゃ', 'ない', 'か', '」', '「', 'よく', 'よく', '考え', '##り', '##ゃ', '『', '猫', 'も', '[UNK]', 'も', '大学', 'に', '行く', 'よう', 'に', 'なっ', 'た', 'から', '、', '結果', '的', 'に', '就職', 'でき', 'ない', 'ヤ', '##ツ', 'が', '増え', 'てる', 'だけ', '』', 'だろ', 'う', 'よ', '」', 'など', '、', '“', '大学', '全', '入社', '会', '”', 'に', '懐疑', '的', 'な', '見方', 'を', '示し', 'た', '。', 'さらに', '「', '大学', 'に', 'なんか', '入ら', 'なく', 'たっ', '##て', '、', '『', '一流', '』', 'と', '呼ば', 'れる', '仕事', '人', 'は', 'たくさん', 'いる', '。', '大工', 'に', 'なっ', 'たっ', '##て', 'いい', 'し', '、', 'コック', 'や', '寿司', '職人', 'に', 'なっ', 'たっ', '##て', 'いい', '」', 'と', '、', '若者', 'の', '大学', '進学', '以外', 'の', '選択肢', 'を', '提示', 'し', 'た', '。', 'この', 'メル', '##マ', '##ガ', 'に対して', '、', 'ネット', '掲示板', 'で', 'は', '「', '正', '##論', '」', '「', '大', '##卒', 'バカ', 'が', '増え', 'た', 'の', 'は', '同意', '」', '「', 'よく', '言っ', 'た', '」', 'など', 'たけし', 'の', '意見', 'に', '肯定', '的', 'な', '反応', 'が', '目立っ', 'た', 'が', '、', 'その', '一方', 'で', '「', '実際', 'に', 'は', '四', '大', '##卒', 'が', '応募', '最低', '条件', 'の', '会社', 'が', '多い', '」', '「', '高', '##卒', 'で', 'は', '就職', 'が', 'もっと', '厳しい', '」', '「', '一', '財産', '成し', 'て', '世界', '的', 'に', 'も', '成功', 'が', '認め', 'られ', 'てる', '人', 'が', '言う', 'から', '許さ', 'れる', '」', 'など', '、', 'たけし', 'の', '意見', 'は', '「', '大', '##卒', 'が', '有利', 'で', 'ある', '」', 'という', '就職', '活動', 'の', '現状', 'から', 'は', '程', '##遠', '##い', '“', '理想', '論', '”', 'だ', 'という', '批判', 'も', '見', 'られ', 'た', '。', '【', '関連', '記事', '】', '・', 'ビート', '##た', '##けし', '「', '大', '##卒', '就職', '率', 'の', '低下', 'は', 'バカ', '大学', 'が', '増え', 'た', 'から', '」', '【', '関連', '情報', '】', '・', '【', '話題', '】', 'ビート', '##た', '##けし', '「', '大', '##卒', '就職', '率', 'の', '低下', 'は', 'バカ', '大学', 'が', '増え', 'た', 'から', '」']
    >>> network.tokenizer(dataloader_train.dataset[0][0])
    {'input_ids': [2, 12563, 28447, 24273, 5, 3615, 7, 24956, 464, 28963, 25, 37, 788, 32, 2393, 5, 36, 4986, 28523, 28668, 14607, 6097, 9960, 38, 7, 1902, 26, 20, 10, 2622, 36, 12563, 28447, 24273, 5, 744, 29323, 5, 110, 28708, 8087, 38, 12, 6, 12563, 28447, 24273, 14, 36, 42, 29202, 7460, 786, 5, 3813, 9, 12097, 396, 14, 3731, 10, 40, 38, 64, 6, 2399, 5, 2203, 396, 887, 1964, 11, 1545, 15, 6, 1920, 13129, 12, 15137, 11, 3769, 12, 33, 8, 16178, 9, 69, 4986, 28523, 28668, 12, 42, 29202, 2043, 5, 7460, 786, 5, 3813, 7, 6068, 895, 6, 36, 10639, 396, 7, 3488, 727, 5, 31, 1056, 28659, 28485, 14, 8657, 687, 33, 5, 29, 6172, 45, 18, 1058, 4847, 80, 29, 38, 36, 1755, 1755, 680, 28477, 29360, 63, 6040, 28, 1, 28, 396, 7, 3488, 124, 7, 58, 10, 40, 6, 854, 81, 7, 7460, 203, 80, 1056, 28659, 14, 3731, 7134, 687, 65, 3635, 205, 54, 38, 64, 6, 2203, 396, 151, 3944, 136, 1964, 7, 19971, 81, 18, 8565, 11, 2374, 10, 8, 604, 36, 396, 7, 26198, 14624, 332, 17492, 28456, 6, 63, 15041, 65, 13, 501, 62, 2198, 53, 9, 12959, 33, 8, 23241, 7, 58, 17492, 28456, 2575, 15, 6, 10452, 49, 15625, 8737, 7, 58, 17492, 28456, 2575, 38, 13, 6, 6741, 5, 396, 3227, 1000, 5, 12721, 11, 6220, 15, 10, 8, 70, 4986, 28523, 28668, 769, 6, 1920, 13129, 12, 9, 36, 371, 28963, 38, 36, 42, 29202, 12097, 14, 3731, 10, 5, 9, 7709, 38, 36, 1755, 3083, 10, 38, 64, 16178, 5, 3149, 7, 11720, 81, 18, 2512, 14, 12058, 10, 14, 6, 59, 862, 12, 36, 1379, 7, 9, 755, 42, 29202, 14, 7290, 5292, 1977, 5, 811, 14, 707, 38, 36, 107, 29202, 12, 9, 7460, 14, 8065, 6484, 38, 36, 52, 6062, 8638, 16, 324, 81, 7, 28, 1320, 14, 1495, 84, 7134, 53, 14, 2217, 40, 5440, 62, 38, 64, 6, 16178, 5, 3149, 9, 36, 42, 29202, 14, 8230, 12, 31, 38, 140, 7460, 455, 5, 8689, 40, 9, 1404, 29393, 28457, 2203, 7619, 887, 1964, 75, 140, 1941, 28, 212, 84, 10, 8, 9680, 1634, 2622, 9594, 35, 12563, 28447, 24273, 36, 42, 29202, 7460, 786, 5, 3813, 9, 12097, 396, 14, 3731, 10, 40, 38, 9680, 1634, 933, 9594, 35, 9680, 4459, 9594, 12563, 28447, 24273, 36, 42, 29202, 7460, 786, 5, 3813, 9, 12097, 396, 14, 3731, 10, 40, 38, 3], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    """

    # dataloader
    dataloader_train = LivedoorNewsDataLoader(
        network.tokenizer, tokenizer_params_input={"padding": True, "max_length": network.tokenizer.model_max_length, "truncation": True}, aftprocs=[lambda x, y: [dict(x), y]],
        root='./data', train=True,  download=True, batch_size=64, shuffle=True,  num_workers=4
    )
    dataloader_valid = LivedoorNewsDataLoader(
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