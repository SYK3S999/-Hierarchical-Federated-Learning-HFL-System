import torch
import torchvision
import pickle
from torch.utils.data import Subset
import numpy as np

def prepare_mnist_subset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Training subset: 400 images, 100 per client, non-IID
    labels = torch.tensor(trainset.targets)
    idx = torch.arange(len(labels))
    client_data = []
    for i in range(4):
        digit_range = list(range(i * 3, (i + 1) * 3))  # 0-2, 3-5, 6-8, 9-0-1
        mask = torch.isin(labels, torch.tensor(digit_range) % 10)
        client_idx = idx[mask][:100]
        client_data.append(Subset(trainset, client_idx))

    # Test subset: 100 images
    test_idx = torch.arange(100)
    test_subset = Subset(testset, test_idx)

    # Save
    with open('data/mnist_train_400.pkl', 'wb') as f:
        pickle.dump(client_data, f)
    with open('data/mnist_test_100.pkl', 'wb') as f:
        pickle.dump(test_subset, f)

if __name__ == "__main__":
    prepare_mnist_subset()
    print("MNIST subsets prepared: mnist_train_400.pkl, mnist_test_100.pkl")