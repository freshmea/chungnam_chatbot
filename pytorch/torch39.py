# 1 bert library import
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# 2 load dataset
train_df = pd.read_csv("data/training.txt", sep="\t")
valid_df = pd.read_csv("data/validing.txt", sep="\t")
test_df = pd.read_csv("data/testing.txt", sep="\t")


# 3 df sample
train_df = train_df.sample(frac=0.1, random_state=500)
valid_df = valid_df.sample(frac=0.1, random_state=500)
test_df = test_df.sample(frac=0.1, random_state=500)


# 4 dataset class
class BertDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return sentence, label


# 5 dataset object, loader object
train_dataset = BertDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

valid_dataset = BertDataset(valid_df)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True, num_workers=0)

test_dataset = BertDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)

# 6 tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model.to(device)


# 7 optimized model save
def save_checkpoint(save_path, model, valid_loss):
    if not save_path:
        return
    state_dict = {"model_state_dict": model.state_dict(), "valid_loss": valid_loss}
    torch.save(state_dict, save_path)
    print(f"Model saved to ==> {save_path}")


def load_checkpoint(load_path, model):
    if not load_path:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f"Model loaded from <== {load_path}")
    model.load_state_dict(state_dict["model_state_dict"])
    return state_dict["valid_loss"]


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if not save_path:
        return
    state_dict = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "global_steps_list": global_steps_list,
    }
    torch.save(state_dict, save_path)
    print(f"Model saved to ==> {save_path}")


def load_metrics(load_path):
    if not load_path:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f"Model loaded from <== {load_path}")
    return (
        state_dict["train_loss_list"],
        state_dict["valid_loss_list"],
        state_dict["global_steps_list"],
    )
