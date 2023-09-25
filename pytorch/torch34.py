# 1 
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# 2
df = pd.read_csv('data/diabetes.csv')
print(df.head())
X = df[df.columns[:-1]]
y = df[df.columns[-1]]
X = X.values
y = torch.tensor(y.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# 3
ms = MinMaxScaler()
ss = StandardScaler()

X_train = ms.fit_transform(X_train)
X_test = ms.fit_transform(X_test)
y_train = ss.fit_transform(y_train.reshape(-1, 1))
y_test = ss.fit_transform(y_test.reshape(-1, 1))


# 4 
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


# 5
train_data = CustomDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
test_data = CustomDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 6
class BinaryClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(8, 64, bias=True)
        self.layer_2 = nn.Linear(64, 64, bias=True)
        self.out_layer = nn.Linear(64, 1, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
    
    def forward(self, input):
        x = self.relu(self.layer_1(input))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        return x 
