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
import numpy as np

# from tqdm import tqdm
from tqdm.notebook import tqdm_notebook

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
    root="pytorch/data/archive/train", transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root="pytorch/data/archive/test", transform=val_transform
)
train_dataloder = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataset = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


# 4
def resnet50(pretrained=True, requires_grad=False):
    model = models.resnet50(progress=True, pretrained=pretrained)
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False
    elif requires_grad:
        for param in model.parameters():
            param.requires_grad = True
    model.fc = nn.Linear(2048, 2)
    return model


# 5 learning rate scheduler
class LRScheduler:
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            min_lr=self.min_lr,
            factor=self.factor,
            verbose=True,
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


# 6 early stopping
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path="data/checkpoint.pt"):
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f}")
            print(f"--> {val_loss:.6f}).  Saving model ...")
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss


# 7 parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--lr-scheduler", dest="lr_scheduler", action="store_true", default=False
)
parser.add_argument(
    "--early-stopping", dest="early_stopping", action="store_true", default=False
)
# args = vars(parser.parse_args())


# 8
print(f"Computation device: {device}\n")
model = models.resnet50(pretrained=True).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters\n")


# 9 training
def training(model, train_dataloader, train_dataset, optimizer, criterion):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm_notebook(train_dataloader, desc="Training", leave=True)

    for i, data in prog_bar:
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    train_accuracy = 100.0 * train_running_correct / total
    return train_loss, train_accuracy
