import torchvision.transforms as transforms

mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,),(1.0,))])

from torchvision.datasets import MNIST
import requests
download_root = 'pythorch/data/MNIST_DATASET'

train_dataset = MNIST(download_root, transform=mnist_transform, 
                      train=True, download=True)
valid_dataset = MNIST(download_root, transform=mnist_transform, 
                      train=False, download=True)
test_dataset = MNIST(download_root, transform=mnist_transform, 
                      train=False, download=True)
