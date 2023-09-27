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


# 8 model training function
def train(
    model,
    optimizer,
    criterion=nn.BCELoss(),
    num_epoch=5,
    eval_every=len(train_loader) // 2,
    best_valid_loss=float("Inf"),
):
    total_correct = 0.0
    total_len = 0.0
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    model.train()
    for epoch in range(num_epoch):
        for text, label in train_loader:
            optimizer.zero_grad()
            encode_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
            padded_list = [e + [0] * (512 - len(e)) for e in encode_list]
            sample = torch.tensor(padded_list)
            sample, label = sample.to(device), label.to(device)
            labels = torch.tensor(label)
            outputs = model(sample, labels=labels)
            loss, logits = outputs

            # loss function
            pred = torch.argmax(F.softmax(logits), dim=1)
            correct = pred.eq(labels)
            total_correct += correct.sum().item()
            total_len += len(labels)
            running_loss = loss.item()

            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    for text, label in valid_loader:
                        encode_list = [
                            tokenizer.encode(t, add_special_tokens=True) for t in text
                        ]
                        padded_list = [e + [0] * (512 - len(e)) for e in encode_list]
                        sample = torch.tensor(padded_list)
                        sample, label = sample.to(device), label.to(device)
                        labels = torch.tensor(label)
                        outputs = model(sample, labels=labels)
                        loss, logits = outputs
                        valid_running_loss += loss.item()

                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                print(
                    f"Epoch [{epoch+1}/{num_epoch}], Step [{global_step}/{num_epoch*len(train_loader)}], Train Loss: {average_train_loss:.4f}, Valid Loss: {average_valid_loss:.4f}"
                )

                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint("data/model.pt", model, best_valid_loss)
                    save_metrics(
                        "data/metrics.pt",
                        train_loss_list,
                        valid_loss_list,
                        global_steps_list,
                    )
    save_metrics("data/metrics.pt", train_loss_list, valid_loss_list, global_steps_list)
    print("훈련 종료")


# 9 train model
optimizer = optim.Adam(model.parameters(), lr=2e-5)
train(model=model, optimizer=optimizer)
