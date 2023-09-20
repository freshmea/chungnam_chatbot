# 1번 셀
import torch
import torchtext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import string
import random

# 2번 셀
start = time.time()
TEXT = torchtext.data.Field(lower=True, batch_first=False, fix_length=200)
LABEL = torchtext.data.Field(sequential=False)

# 3번 셀
train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL)

# 4번 셀
print(vars(train_data.examples[0]))

# 5번 셀

for example in train_data.examples:
    text = [x.lower() for x in vars(example)["text"]]
    text = [x.replace("<br", "") for x in text]
    text = ["".join(c for c in s if c not in string.punctuation) for s in text]
    text = [s for s in text if s]
    vars(example)["text"] = text

# 6번 셀

train_data, valid_data = train_data.split(random_state=random.seed(0), split_ratio=0.8)

# 7번 셀
print("Number of training example: ", len(train_data))
print("Number of validating example: ", len(valid_data))
print("Number of testing examples: ", len(test_data))

# 8번 셀
TEXT.build_vocab(train_data, max_size=10000, min_freq=10, vectors=None)
LABEL.build_vocab(train_data)

print("TEXT tokens Vocabulary size: ", len(TEXT.vocab))
print("Label tokens size: ", len(LABEL.vocab))
print(LABEL.vocab.stoi)

# 9번 셀
BATCH_SIZE = [64 for _ in range(100)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embeding_dim = 100
hidden_size = 300

train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, test_data), batch_sizes=BATCH_SIZE, device=device
)


# 10번 셀
class RNNCell_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.rnn = nn.RNNCell(input_dim, hidden_size)

    def forward(self, inputs):
        bz = inputs.shape[1]
        ht = torch.zeros(bz, hidden_size, device=device)
        for word in inputs:
            ht = self.rnn(word, ht)
        return ht


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.em = nn.Embedding(len(TEXT.vocab.stoi), embeding_dim)
        self.rnn = RNNCell_Encoder(embeding_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.em(x)
        x = self.rnn(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 11번 셀
model = Net()
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 12번 셀
def training(epoch, model, trainloader, validloader):
    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for b in trainloader:
        x, y = b.text, b.label
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            predicted = torch.argmax(y_pred, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = correct / total

    valid_correct = 0
    valid_total = 0
    valid_running_loss = 0

    model.eval()
    with torch.no_grad():
        for b in validloader:
            x, y = b.text, b.label
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            predicted = torch.argmax(y_pred, dim=1)
            valid_correct += (predicted == y).sum().item()
            valid_total += y.size(0)
            valid_running_loss += loss.item()

    epoch_valid_loss = valid_running_loss / len(validloader)
    epoch_valid_acc = valid_correct / valid_total

    print(
        f"epoch: {epoch}",
        f"train loss: {epoch_loss:.4f}",
        f"train acc: {epoch_acc:.4f}",
        f"valid loss: {epoch_valid_loss:.4f}",
        f"valid acc: {epoch_valid_acc:.4f}",
    )
    return epoch_loss, epoch_acc, epoch_valid_loss, epoch_valid_acc


# 13 번 셀
epochs = 5
train_loss = []
train_acc = []
valid_loss = []
valid_acc = []

start = time.time()
for epoch in range(epochs):
    loss, acc, v_loss, v_acc = training(epoch, model, train_iter, valid_iter)
    train_loss.append(loss)
    train_acc.append(acc)
    valid_loss.append(v_loss)
    valid_acc.append(v_acc)

end = time.time()
print(end - start)


# 14번 셀
def evaluate(epoch, model, testloader):
    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for b in testloader:
            x, y = b.text, b.label
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            predicted = torch.argmax(y_pred, dim=1)
            test_correct += (predicted == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader)
    epoch_test_acc = test_correct / test_total

    print(
        f"epoch: {epoch}",
        f"test loss: {epoch_test_loss:.4f}",
        f"test acc: {epoch_test_acc:.4f}",
    )
    return epoch_test_loss, epoch_test_acc


# 15번 셀
start = time.time()
epochs = 5
test_loss = []
test_acc = []

for epoch in range(epoch):
    loss, acc = evaluate(epoch, model, test_iter)
    test_loss.append(loss)
    test_acc.append(acc)

end = time.time()
print(end - start)
