# 1
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm_notebook

# from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2
data = pd.read_csv("data/SBUX.csv")
print(data.dtypes)

# 3
data["Date"] = pd.to_datetime(data["Date"])
# data = data.set_index("Date")
data.set_index("Date", inplace=True)
data["Volume"] = data["Volume"].astype(float)
print(data.dtypes)


# 4
X = data.iloc[:, :-1]
y = data.iloc[:, 5:6]
print(X)
print(y)

# 5
ms = MinMaxScaler()
ss = StandardScaler()

X_ss = ss.fit_transform(X)
y_ms = ms.fit_transform(y)

X_train = X_ss[:200, :]
X_test = X_ss[200:, :]

y_train = y_ms[:200, :]
y_test = y_ms[200:, :]

print("Training Shape: ", X_train.shape, X_test.shape)
print("Testing Shape: ", y_train.shape, y_test.shape)

# 6
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))
print("Training Shape", X_train_tensors.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors.shape, y_test_tensors.shape)
X_train_tensors_f = torch.reshape(
    X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1])
)
X_test_tensors_f = torch.reshape(
    X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])
)

print("Training Shape", X_train_tensors_f.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_f.shape, y_test_tensors.shape)


# 7
# class LSTMcell(nn.Module):
# ------
# nn.LSTM


class BiLSTMModule(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(
            torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        )
        c_0 = Variable(
            torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        )
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        return out


# 8
num_epochs = 1000
learning_rate = 0.0001

input_size = 5
hidden_size = 2
num_layers = 1

num_classes = 1
model = BiLSTMModule(
    num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1]
).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 9
for epoch in range(num_epochs):
    outputs = model(X_train_tensors_f.to(device))
    optimizer.zero_grad()
    loss = criterion(outputs, y_train_tensors.to(device))
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, Loss: %1.5f" % (epoch, loss.item()))

# 10
df_x_ss = ss.transform(data.iloc[:, :-1])
df_y_ms = ms.transform(data.iloc[:, -1:])

df_x_ss = torch.Tensor(df_x_ss)
df_y_ms = torch.Tensor(df_y_ms)
df_x_ss = torch.reshape(df_x_ss, (df_x_ss.shape[0], 1, df_x_ss.shape[1]))

# 11
train_predict = model(df_x_ss.to(device))
predicted = train_predict.cpu().data.numpy()
label_y = df_y_ms.data.numpy()

predicted = ms.inverse_transform(predicted)
label_y = ms.inverse_transform(label_y)
plt.figure(figsize=(10, 6))
plt.axvline(x=200, c="r", linestyle="--")

plt.plot(label_y, label="Actual Data")
plt.plot(predicted, label="Predicted Data")
plt.title("SBUX Stock Price Prediction")
plt.legend()
plt.show()
