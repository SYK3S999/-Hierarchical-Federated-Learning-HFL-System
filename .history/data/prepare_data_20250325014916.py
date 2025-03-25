import torch
from torchvision import datasets, transforms
import pickle
import random

transform = transforms.ToTensor()
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 10 clients with varying sizes
sizes = [150, 100, 200, 120, 180, 90, 160, 130, 140, 110]
client_data = []
for i in range(10):
    indices = [idx for idx, (_, label) in enumerate(trainset) if label % 10 == i]
    random.shuffle(indices)
    client_data.append(torch.utils.data.Subset(trainset, indices[:sizes[i]]))
with open('data/mnist_train_variable.pkl', 'wb') as f:
    pickle.dump(client_data, f)

# Test set (1000 images)
test_indices = random.sample(range(len(testset)), 1000)
test_subset = torch.utils.data.Subset(testset, test_indices)
with open('data/mnist_test_1000.pkl', 'wb') as f:
    pickle.dump(test_subset, f)