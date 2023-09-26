# 1
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# 2
df = pd.read_csv("data/diabetes.csv")
print(df.head())
X = df[df.columns[:-1]]
y = df[df.columns[-1]]
X = X.values
y = torch.tensor(y.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

ms = MinMaxScaler()
ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ms.fit_transform(y_train)
y_test = ms.fit_transform(y_test)


class customdataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


train_data = customdataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
test_data = customdataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)


class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        self.layer_1 = nn.Linear(8, 64, bias=True)
        self.layer_2 = nn.Linear(64, 64, bias=True)
        self.layer_out = nn.Linear(64, 1, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x


epochs = 1000 + 1
print_epoch = 100
LEARNING_RATE = 1e-2

model = binaryClassification()
model.to(device)
print(model)
BCE = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


for epoch in range(epochs):
    iteration_loss = 0.0
    iteration_accuracy = 0.0

    model.train()
    for i, data in enumerate(train_loader):
        X, y = data
        X, y = X.to(device), y.to(device)
        y_pred = model(X.float()).to(device)
        loss = BCE(y_pred, y.reshape(-1, 1).float())

        iteration_loss += loss
        iteration_accuracy += accuracy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % print_epoch == 0:
        print(
            "Train: epoch: {0} - loss: {1:.5f}; acc: {2:.3f}".format(
                epoch, iteration_loss / (i + 1), iteration_accuracy / (i + 1)
            )
        )

    iteration_loss = 0.0
    iteration_accuracy = 0.0
    model.eval()
    for i, data in enumerate(test_loader):
        X, y = data
        X, y = X.to(device), y.to(device)
        y_pred = model(X.float()).to(device)
        loss = BCE(y_pred, y.reshape(-1, 1).float())
        iteration_loss += loss
        iteration_accuracy += accuracy(y_pred, y)
    if epoch % print_epoch == 0:
        print(
            "Test: epoch: {0} - loss: {1:.5f}; acc: {2:.3f}".format(
                epoch, iteration_loss / (i + 1), iteration_accuracy / (i + 1)
            )
        )
