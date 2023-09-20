# 1번 셀
import torch
import torchtext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

# 2번 셀
start = time.time()
TEXT = torchtext.data.Field(lower=True, batch_first=False, fix_length=200)
LABEL = torchtext.data.Field(sequential=False)

# 3번 셀
train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL)

# 4번 셀
print(vars(train_data.examples[0]))

# 5번 셀
