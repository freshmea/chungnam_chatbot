import os, time, copy, glob, cv2, shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

data_path = 'pytorch/data/catanddog/train'

transform = transforms.Compose(
    [
        transforms.Resize([256,256]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
)

train_dataset = torchvision.datasets.ImageFolder(
    data_path, transform=transform,
)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True
)

print(len(train_dataset))

# samples, labels = train_loader._get_iterator()._next_data()
# classes = {0:'cat', 1:'dog'}
# fig = plt.figure(figsize=(16,24))
# for i in range(24):
#     fig.add_subplot(4, 6, i+1)
#     plt.title(classes[labels[i].item()])
#     plt.axis('off')
#     plt.imshow(np.transpose(samples[i].numpy(), (1,2,0)))
# plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)
# plt.show()

