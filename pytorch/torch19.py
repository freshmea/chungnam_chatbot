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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "enable"

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


def train_model(
    model, dataloaders, criterion, optimizer, device, num_epochs=13, is_train=True
):
    since = time.time()
    acc_history = []
    loss_history = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        # print('----------')
        print("-" * 10)
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders:
            inputs, labels = inputs.to(device), labels.to(device)

            model.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc

        acc_history.append(epoch_acc.item())
        loss_history.append(epoch_loss)
        torch.save(
            model.state.dict(), os.path.join(data_path, "0:0=2d.pth".format(epoch))
        )
        print()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Acc: {best_acc:4f}")
    return acc_history, loss_history


params_to_update = []
for name, param in resnet18.named_parameters():
    if param.requires_grad is True:
        params_to_update.append(param)
        print("\t", name)

optimizer = optim.Adam(params_to_update)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
train_acc_hist, train_loss_hist = train_model(
    resnet18, train_loader, criterion, optimizer, device
)
