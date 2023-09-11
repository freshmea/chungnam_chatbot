import copy
import glob
import os
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

data_path = "C:/chungnam_chatbot/pytorch/data/catanddog/train"

transform = transforms.Compose(
    [
        transforms.Resize([256, 256]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

train_dataset = torchvision.datasets.ImageFolder(
    data_path,
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


print(len(train_dataset))

# samples, labels = train_loader._get_iterator()._next_data()
# classes = {0: "cat", 1: "dog"}
# fig = plt.figure(figsize=(16, 24))
# for i in range(24):
#     fig.add_subplot(4, 6, i + 1)
#     plt.title(classes[labels[i].item()])
#     plt.axis("off")
#     plt.imshow(np.transpose(samples[i].numpy(), (1, 2, 0)))
# plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)
# plt.show()

resnet18 = models.resnet18(pretrained=True)


def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


set_parameter_requires_grad(resnet18)
resnet18.fc = nn.Linear(512, 1)

for name, param in resnet18.named_parameters():
    print(name)
print("requires_grad 가 True 인 layer:-----------")
for name, param in resnet18.named_parameters():
    if param.requires_grad:
        print(name)

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(512, 2)
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.fc.parameters())
cost = torch.nn.CrossEntropyLoss()
print(model)
