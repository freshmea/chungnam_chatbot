import torch
import pandas as pd

data = pd.read_csv("pytorch/test.csv")
# print(data.keys())
torch_data = torch.from_numpy(data["kor"].values)
print(torch_data)

str()
