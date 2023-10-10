# variational autoencoder digit 0-9

# 1 - Import library
import torch
import os
from tensorboardX import SummaryWriter

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2 prepare dataset
transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root="pytorch/data2", train=True, transform=transforms, download=True
)

test_dataset = datasets.MNIST(
    root="pytorch/data2", train=False, transform=transforms, download=True
)

train_loder = DataLoader(
    train_dataset, batch_size=100, shuffle=True, num_workers=4, pin_memory=False
)

test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)


# 3 Network Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input1 = nn.Linear(input_dim, hidden_dim)
        self.input2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True

    def forward(self, x):
        h_ = self.LeakyReLU(self.input1(x))
        h_ = self.LeakyReLU(self.input2(h_))
        mean = self.mean(h_)
        log_var = self.var(h_)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.input1 = nn.Linear(latent_dim, hidden_dim)
        self.input2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.input1(x))
        h = self.LeakyReLU(self.input2(h))
        x_hat = torch.sigmoid(self.output(h))
        return x_hat


class Model_network(nn.Module):
    def __init__(self, Encoder, Decoder):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(z)
        return x_hat, mean, log_var


# 4 object initiation
x_dim = 784
hidden_dim = 400
latent_dim = 200
epochs = 30
batch_size = 100

encoder = Encoder(x_dim, hidden_dim, latent_dim).to(device)
decoder = Decoder(latent_dim, hidden_dim, x_dim).to(device)
model = Model_network(encoder, decoder).to(device)


# 5 loss function and optimizer
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss, KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)


# model train function
saved_loc = "scalar/"
writer = SummaryWriter(saved_loc)

model.train()


def train(epoch, model, train_loder, optimizer):
    train_loss = 0
    for batch_idx, (x, _) in enumerate(train_loder):
        x = x.view(batch_size, x_dim)
        x = x.to(device)

        optimizer.zero_grad()
        x_hat, mean, log_var = model(x)
        BCE, KLD = loss_function(x, x_hat, mean, log_var)
        loss = BCE + KLD
        writer.add_scalar(
            "Train/Reconstruction Error",
            BCE.item(),
            batch_idx + epoch * len(train_loder.dataset) / batch_size,
        )
        writer.add_scalar(
            "Train/KL-Divergence",
            KLD.item(),
            batch_idx + epoch * len(train_loder.dataset) / batch_size,
        )
        writer.add_scalar(
            "Train/Total Loss",
            loss.item(),
            batch_idx + epoch * len(train_loder.dataset) / batch_size,
        )
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(x)}/{len(train_loder.dataset)} ({100. * batch_idx / len(train_loder):.0f}%)]\tLoss: {loss.item() / len(x):.6f}"
            )
    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loder.dataset)
        )
    )
