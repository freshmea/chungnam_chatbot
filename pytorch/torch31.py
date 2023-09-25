# 1
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm

matplotlib.style.use("ggplot")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# 3
train_dataset = datasets.ImageFolder(
    root="/data/archive/train", transform=train_transform
)
val_dataset = datasets.ImageFolder(root="/data/archive/val", transform=val_transform)
train_dataloder = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataset = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 4
