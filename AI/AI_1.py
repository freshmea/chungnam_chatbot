import torch
import pandas as pd


a = torch.tensor([[1., -1.],[1.,-1]], dtype=torch.int64, device="cuda:0")
print(a)
print(a.cpu().numpy().dtype)
