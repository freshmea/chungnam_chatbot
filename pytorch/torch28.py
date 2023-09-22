# 1
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# import torchvision.datasets as dataset
from torch.autograd import Variable

# from torch.nn import Parameter
# from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from torchvision.datasets import MNIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 2
mnist_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]
)


download_root = "data/MNIST_DATASET"

train_dataset = MNIST(
    download_root, transform=mnist_transform, train=True, download=True
)
valid_dataset = MNIST(
    download_root, transform=mnist_transform, train=False, download=True
)
test_dataset = MNIST(
    download_root, transform=mnist_transform, train=False, download=True
)


# 3
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 4
batch_size = 100
n_iters = 6000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)


# 5.
# nn.GRUCell


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)

        hy = newgate + inputgate * (hidden - newgate)
        return hy


# 6
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = Variable(
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        )
        outs = []
        hn = h0[0, :, :]
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self.fc(out)
        return out


# 7
input_dim = 28
hidden_dim = 128
layer_dim = 1
output_dim = 10

model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 8
seq_dim = 28
loss_list = []
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        images = Variable(images.view(-1, seq_dim, input_dim).to(device))
        labels = Variable(labels.to(device))
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels).to(device)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data)
        iter += 1
        if iter % 500 == 0:
            model.eval()
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = Variable(images.view(-1, seq_dim, input_dim).to(device))
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels.cpu()).sum()
            accuracy = 100 * correct / total
            loss_list.append(loss.data)
            print(f"Iteration: {iter}. Loss: {loss.data}. Accuracy: {accuracy}")


# 9
def evaluate(model, val_iter):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    for images, labels in val_iter:
        images = Variable(images.view(-1, seq_dim, input_dim).to(device))
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels.to(device), reduction="sum")
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        total_loss += loss.item()
        correct += (predicted.cpu() == labels.cpu()).sum()

    avg_loss = total_loss / len(val_iter.dataset)
    avg_acc = correct / total
    return avg_loss, avg_acc


test_loss, test_acc = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:5.2f}. Test Accuracy: {test_acc:5.2f}")
