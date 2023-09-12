# 1 번 셀 --------------------------------
# import shutil
# import glob
# from torchvision.transforms import ToTensor
# import torchvision.models as models
# from tqdm.notebook import tqdm_notebook
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from torchsummary import summary

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import random
from PIL import Image
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2 번 셀 --------------------------------
class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


# 3 번 셀 --------------------------------
cat_directory = r"C:\chungnam_chatbot\pytorch\data\dogs-vs-cats\Cat"
dog_directory = r"C:\chungnam_chatbot\pytorch\data\dogs-vs-cats\Dog"

cat_images_filepaths = sorted(
    [os.path.join(cat_directory, f) for f in os.listdir(cat_directory)]
)
dog_images_filepaths = sorted(
    [os.path.join(dog_directory, f) for f in os.listdir(dog_directory)]
)

# [[], []] -> [....]
# cat_images_filepaths.extend(dog_images_filepaths)
images_filepaths = [*cat_images_filepaths, *dog_images_filepaths]

# 에러 처리.
correct_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]

random.seed(42)
random.shuffle(correct_images_filepaths)
train_image_filepaths = correct_images_filepaths[:400]
val_image_filepaths = correct_images_filepaths[400:-10]
test_image_filepaths = correct_images_filepaths[-10:]
print(len(train_image_filepaths), len(val_image_filepaths), len(test_image_filepaths))


# 4 번 셀 --------------------------------
def display_image_grid(images_filepaths, predicted_labels=(), cols=5):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
        predicted_label = predicted_labels[i] if predicted_labels else true_label
        color = "green" if true_label == predicted_label else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


display_image_grid(test_image_filepaths)


# 5 번 셀 --------------------------------
class DogvsCatDataset(Dataset):
    def __init__(self, file_list, tranform=None, phase="train") -> None:
        super().__init__()
        self.file_list = file_list
        self.transform = tranform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        # aa
        label = img_path.split("\\")[-1].split(".")[0]
        if label == "dog":
            label = 1
        elif label == "cat":
            label = 0
        return img_transformed, label


# 6 번 셀 --------------------------------
size = 256
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 32

train_dataset = DogvsCatDataset(
    train_image_filepaths, tranform=ImageTransform(size, mean, std), phase="train"
)

val_dataset = DogvsCatDataset(
    val_image_filepaths, tranform=ImageTransform(size, mean, std), phase="val"
)
test_dataset = DogvsCatDataset(
    val_image_filepaths, tranform=ImageTransform(size, mean, std), phase="val"
)
index = 0
# print(train_dataset.__getitem__(index)[0].size())
# print(train_dataset.__getitem__(index)[1])
print(train_dataset[index][0].size())
print(train_dataset[index][1])


# 7 번 셀 --------------------------------
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

batch_iterator = iter(train_dataloader)
inputs, label = next(batch_iterator)
print(inputs.size())
print(label)


# 8 번 셀 --------------------------------
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 9 번 셀 --------------------------------
model = AlexNet()
model.to(device)
print(model)

# 10 번 셀 --------------------------------

# mat 갯수가 달라지는 애러 발생.
# summary(model, input_size=(3, 244, 244))


# 11 번 셀 --------------------------------
def count_parameter(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameter(model):,} trainable parameters")

# 12 번 셀 ---------------------------------
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

model.to(device)
criterion = criterion.to(device)


def train_model(model, dataloader_dict, criterion, optimizer, num_epoch):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epoch):
        print(f"Epoch {epoch+1} / {num_epoch}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            # tqdm 실행 안됨.
            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s")
    return model


num_epoch = 10
model = train_model(model, dataloader_dict, criterion, optimizer, num_epoch)


# 13 번 셀 ---------------------------------
id_list = []
pred_list = []
_id = 0

with torch.no_grad():
    for test_path in tqdm(test_image_filepaths):
        img = Image.open(test_path)
        _id = test_path.split("\\")[-1].split(".")[1]
        transform = ImageTransform(size, mean, std)
        img = transform(img, phase="val")
        img = img.unsqueeze(0)
        img = img.to(device)

        model.eval()
        outputs = model(img)
        preds = F.softmax(outputs, dim=1)[:, 1].tolist()
        id_list.append(_id)
        pred_list.append(preds[0])

res = pd.DataFrame({"id": id_list, "label": pred_list})

res.sort_values(by="id", inplace=True)
res.reset_index(drop=True, inplace=True)

res.to_csv(r"C:\chungnam_chatbot\pytorch\data", index=False)

res.head(10)

# 14 번 셀 ---------------------------------
class_ = Classes = {0: "cat", 1: "dog"}


def display_image_grid2(images_filepaths, predicted_labels=(), cols=5):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        a = random.choice(res["id"].values)
        label = res.loc[res["id"] == a, "label"].values[0]
        if label > 0.5:
            label = 1
        else:
            label = 0
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(class_[label])
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


display_image_grid2(test_image_filepaths)
