# 1 import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from kmeans_pytorch import kmeans, kmeans_predict


# 2 load data
df = pd.read_csv("data/iris.csv")
df.info()
print("-" * 30)
print(df)
