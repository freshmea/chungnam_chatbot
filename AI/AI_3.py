import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataSet(Dataset):
    def __init__(self, csv_file, nColums):
        self.label = pd.read_csv(csv_file)
        self.nColums = nColums
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample = torch.tensor(self.label.iloc[idx,0:self.nColums]).int()
        label = torch.tensor(self.label.iloc[idx,self.nColums]).int()
        return sample, label

tensor_dataset = CustomDataSet(r'chap02/data/covtype.csv', 11)
dataset = DataLoader(tensor_dataset, batch_size=40, shuffle=True)

for i, data in enumerate(dataset, 0):
    print(data)
    print(i, end='')
    batch=data[0]
    print(batch.size())