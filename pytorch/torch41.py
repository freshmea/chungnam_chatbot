# 1 import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from kmeans_pytorch import kmeans, kmeans_predict

from pytorch.torch14 import X

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2 load data
df = pd.read_csv("data/iris.csv")
df.info()
print("-" * 30)
print(df)


# 3 word embedding
data = pd.get_dummies(df, columns=["Species"])
data["Species_Iris-setosa"] = data["Species_Iris-setosa"].astype("float32")
data["Species_Iris-versicolor"] = data["Species_Iris-versicolor"].astype("float32")
data["Species_Iris-virginica"] = data["Species_Iris-virginica"].astype("float32")
data.info()


# 4 split data
x, y = train_test_split(data, test_size=0.2, random_state=123)
print(type(x), "\n", x)
print(type(y), "\n", y)


# 5 scaling
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_scaled = ss.fit(data).transform(x)
y_scaled = ss.fit(data).transform(y)
print(type(x_scaled), "\n", x_scaled)

# 6 to tensor
x = torch.from_numpy(x_scaled).to(device)
y = torch.from_numpy(y_scaled).to(device)
print(type(x), "\n", x.size(), "\n", y.size(), x)
