# 1번 셀
import torch
import torchtext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import string
import random

# 2번 셀
start = time.time()
TEXT = torchtext.data.Field(lower=True, batch_first=False, fix_length=200)
LABEL = torchtext.data.Field(sequential=False)

# 3번 셀
train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL)

# 4번 셀
print(vars(train_data.examples[0]))

# 5번 셀

for example in train_data.examples:
    text = [x.lower() for x in vars(example)["text"]]
    text = [x.replace("<br", "") for x in text]
    text = ["".join(c for c in s if c not in string.punctuation) for s in text]
    text = [s for s in text if s]
    vars(example)["text"] = text

# 6번 셀

train_data, valid_data = train_data.split(random_state=random.seed(0), split_ratio=0.8)
