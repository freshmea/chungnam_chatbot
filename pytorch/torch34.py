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