import torch
from torchvision import datasets, transforms
import pickle
import random

transform = transforms.ToTensor()
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 10 clients with varying sizes and non-IID data
sizes = [100, 150, 50, 120, 80, 130, 70, 110, 90, 140]  # Total: 1040 images
client_data = []
for i in range(10):
    primary_digit = i % 10
    indices = [idx for idx, (_, label) in enumerate(trainset) if label == primary_digit][:int(sizes[i] * 0.8)]
    other_indices = [idx for idx, (_, label) in enumerate(trainset) if label != primary_digit][:int(sizes[i] * 0.2)]
    random.shuffle(indices)
    random.shuffle(other_indices)
    client_data.append(torch.utils.data.Subset(trainset, indices + other_indices[:sizes[i] - len(indices)]))

# Save training data for 10 clients
with open('data/mnist_train_10clients.pkl', 'wb') as f:
    pickle.dump(client_data, f)

# Test set (100 images, matching your original setup)
test_indices = random.sample(range(len(testset)), 100)
with open('data/mnist_test_100.pkl', 'wb') as f:
    pickle.dump(torch.utils.data.Subset(testset, test_indices), f)