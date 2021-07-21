import random, os
import zipfile, tarfile
from typing import Callable
import pandas as pd
import numpy as np

import torch, torchvision
import torchvision.transforms as transforms

from kktorch.data.dataset import DataframeDataset
from kktorch.util.com import correct_dirpath, makedirs
from kktorch.util.files import download_file
from kktorch.util.dataframe import text_files_to_dataframe
from kktorch.data.dataloader.dataloader import BaseDataLoader, TextDataLoader


__all__ = [
    "MNISTDataLoader",
    "NewsPaperDataLoader",
    "LivedoorNewsDataLoader",
]


def split_train_test(index: int, seed: int=0, percent: float=0.8):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    indexes = np.random.permutation(np.arange(index))
    size    = int(indexes.shape[0] * percent)
    indexes_train = indexes[:size ]
    indexes_test  = indexes[ size:]
    return indexes_train, indexes_test


def deploy_files(url: str, dirpath: str, download: bool, extract: str="zip"):
    assert isinstance(extract, str) and extract in ["zip", "gz"]
    makedirs(dirpath, exist_ok=True, remake=False)
    filepath = dirpath + os.path.basename(url)
    if download and not os.path.exists(filepath):
        filepath = download_file(url, filepath=filepath)
        if extract == "zip":
            with zipfile.ZipFile(filepath) as existing_zip:
                existing_zip.extractall(dirpath)
        elif extract == "gz":
            with tarfile.open(filepath, 'r:gz') as t:
                t.extractall(path=dirpath)


class MNISTDataLoader(BaseDataLoader):
    def __init__(
        self, root: str='./data', train: bool=True, download: bool=True, 
        transform=[transforms.ToTensor(), ], 
        dtype_data=torch.float32, dtype_target=torch.long, **kwargs
    ):
        transform = transform if isinstance(transform, list) else []
        transform = transforms.Compose(transform)
        dataset   = torchvision.datasets.MNIST(root=root, train=train, download=download, transform=transform)
        super().__init__(dataset, dtype_data=dtype_data, dtype_target=dtype_target, **kwargs)


class NewsPaperDataLoader(TextDataLoader):
    def __init__(
        self, tokenizer: Callable, root: str='./data', train: bool=True, download: bool=True, **kwargs
    ):
        """
        see: https://archive.ics.uci.edu/ml/datasets/News+Aggregator
        CATEGORY News category (b = business, t = science and technology, e = entertainment, m = health)
        replace category name to {"b": 0, "t": 1, "e": 2, "m": 3}
        ex)
            text : 'Fed official says weak data caused by weather, should not slow taper'
            label: 'b'
        """
        self.dirpath = correct_dirpath(root) + "NewsPaper/"
        url          = "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip"
        csv_filepath = self.dirpath + "newsCorpora.csv"
        deploy_files(url, self.dirpath, download, extract="zip")
        df    = pd.read_csv(csv_filepath, sep="\t", header=None)
        df[4] = df[4].map({"b": 0, "t": 1, "e": 2, "m": 3}).astype(np.int32)
        df    = df.loc[np.random.permutation(df.index.values)]
        indexes_train, indexes_test = split_train_test(df.shape[0])
        df_train = df.iloc[indexes_train].copy()
        df_test  = df.iloc[indexes_test ].copy()
        dataset  = DataframeDataset(df_train if train else df_test, columns=[1, 4])
        super().__init__(dataset, dtype_data=torch.long, dtype_target=torch.long, tokenizer=tokenizer, **kwargs)


class LivedoorNewsDataLoader(TextDataLoader):
    def __init__(
        self, tokenizer: Callable, root: str='./data', train: bool=True, download: bool=True, columns=["text", "label"], **kwargs
    ):
        """
        see: https://www.rondhuit.com/download.html#ldcc
        label: {
            'dokujo-tsushin': 0, 'it-life-hack': 1, 'kaden-channel': 2,
            'livedoor-homme': 3, 'movie-enter':4, 'peachy':5, 'smax': 6,
            'sports-watch': 7, 'topic-news': 8
        }
        ex)
            text:  '吉田麻也、サウサンプトンFC移籍が決定\nサッカー日本代表・吉田麻也のサウサンプトンFC移籍が正式に決定した。\n\n同クラブは、公式サイトで「セインツ（サウサンプトンFCの愛称）にとって喜ばしい発表」として、吉田の入団を発表。オフィシャルツイッター（@officialsaints）上でも、「Japanese international Maya Yoshida is set to join #SaintsFC on a three-year deal from VVV-Venlo.」と呟いている。\n\nまた、同ツイッターでは、「プレミアリーグにくることは、子供の頃からの大きな夢の一つ」という吉田のコメントも掲載。オランダ・VVVフェンローからの移籍となった吉田の契約期間は3年となる。\n\n一部には、吉田のサウサンプトンFC入団が決まれば、9月2日のマンU戦でのプレミアデビュー＆香川真司との日本人対決実現という報道もあったが、その可能性も限りなく高まったといえるだろう。日本のファンからすれば、また大きな楽しみが増える吉田のプレミア入りだ。\n'
            site:  sports-watch
            title: '吉田麻也、サウサンプトンFC移籍が決定'
            body:  'サッカー日本代表・吉田麻也のサウサンプトンFC移籍が正式に決定した。\n\n同クラブは、公式サイトで「セインツ（サウサンプトンFCの愛称）にとって喜ばしい発表」として、吉田の入団を発表。オフィシャルツイッター（@officialsaints）上でも、「Japanese international Maya Yoshida is set to join #SaintsFC on a three-year deal from VVV-Venlo.」と呟いている。\n\nまた、同ツイッターでは、「プレミアリーグにくることは、子供の頃からの大きな夢の一つ」という吉田のコメントも掲載。オランダ・VVVフェンローからの移籍となった吉田の契約期間は3年となる。\n\n一部には、吉田のサウサンプトンFC入団が決まれば、9月2日のマンU戦でのプレミアデビュー＆香川真司との日本人対決実現という報道もあったが、その可能性も限りなく高まったといえるだろう。日本のファンからすれば、また大きな楽しみが増える吉田のプレミア入りだ。\n'
        """
        self.dirpath = correct_dirpath(root) + "LivedoorNews/"
        url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
        deploy_files(url, self.dirpath, download, extract="gz")
        filepath_df = self.dirpath + "df.pickle"
        if os.path.exists(filepath_df):
            df = pd.read_pickle(filepath_df)
        else:
            df = text_files_to_dataframe(self.dirpath + "text/", regex_list=[r"[0-9]\.txt$"], encoding="utf8")
            df.to_pickle(filepath_df)
        df["text"]  = df["text"].apply(lambda x: "\n".join(x.split("\n")[2:]))
        df["title"] = df["text"].apply(lambda x: x.split("\n")[0])
        df["body"]  = df["text"].apply(lambda x: "\n".join(x.split("\n")[1:]))
        df["site"]  = df["filename"].apply(lambda x: "-".join(x.split("-")[:-1]))
        df["label"] = df["site"].map({
            'dokujo-tsushin': 0, 'it-life-hack': 1, 'kaden-channel': 2,
            'livedoor-homme': 3, 'movie-enter':4, 'peachy':5, 'smax': 6,
            'sports-watch': 7, 'topic-news': 8
        })
        indexes_train, indexes_test = split_train_test(df.shape[0])
        df_train = df.iloc[indexes_train].copy()
        df_test  = df.iloc[indexes_test ].copy()
        self.df  = df_train if train else df_test
        dataset  = DataframeDataset(df_train if train else df_test, columns=columns)
        super().__init__(dataset, dtype_data=torch.long, dtype_target=torch.long, tokenizer=tokenizer, **kwargs)