# 1번 셀
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
import time
import random
from torchtext.datasets import IMDB

# 2번 셀
start = time.time()
TEXT = torchtext.data.Field(sequential=True, lower=True, batch_first=True)
LABEL = torchtext.data.Field(sequential=False, batch_first=True)

# 3번 셀
train_data, test_data = IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(0), split_ratio=0.8)

TEXT.build_vocab(train_data, max_size=10000, min_freq=10, vectors=None)
LABEL.build_vocab(train_data)

BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4번 셀
train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
)

# 5번 셀
vocab_size = len(TEXT.vocab)
n_classes = 2


# 6번 셀
class BasicRNN(nn.Module):
    def __init__(
        self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2
    ):
        super().__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.RNN(
            embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True
        )
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.rnn(x, h_0)
        h_t = x[:, -1, :]
        self.dropout(h_t)
        logit = torch.sigmoid(self.out(h_t))
        return logit

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()


# 7번 셀
model = BasicRNN(
    n_layers=1,
    hidden_dim=256,
    n_vocab=vocab_size,
    embed_dim=128,
    n_classes=n_classes,
    dropout_p=0.5,
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# 8번 셀
def train(epoch, model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()
        if b % 50 == 0:
            print(
                f"Train Epoch: {epoch} [{b * len(x)}/{len(train_iter.dataset)} ", end=""
            )
            print(f"({len(train_iter.dataset):.0f}%)]\tLoss: {loss.item():.6f}")


# 9번 셀
def evaluate(model, val_iter):
    model.eval()
    corrects, total, total_loss = 0, 0, 0

    for batch in val_iter:
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction="sum")
        total += y.size(0)
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()

    avg_loss = total_loss / len(val_iter.dataset)
    avg_accuracy = corrects / total
    return avg_loss, avg_accuracy


# 10번 셀
BATCH_SIZE = 100
LR = 0.001
EPOCHS = 5
for e in range(1, EPOCHS + 1):
    train(e, model, optimizer, train_iter)
    valid_loss, valid_accuracy = evaluate(model, valid_iter)
    print(f"\n[epoch: {e:3d}] Validation_loss: {valid_loss:4.2f}", end="")
    print(f"| Validation_acc: {valid_accuracy:4.2f}")
