# 1번 셀
import torch
import torchtext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import random
from torchtext.datasets import IMDB

# 2번 셀
start = time.time()
TEXT = torchtext.data.Field(sequential=True, lower=True, batch_first=True)
LABEL = torchtext.data.Field(sequential=False, batch_first=True)

# 3번 셀
train_data, test_data = IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(0), split_ratio=0.8)

TEXT.build_vocab(train_data, max_size=10000, min_freq=10, vectors=None)
LABEL.build_vocab(train_data)

BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4번 셀
train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    shuffle=True,
    repeat=False,
)
