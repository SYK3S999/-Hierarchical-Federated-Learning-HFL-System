import torch
import torch.nn as nn
import torch.optim as optim
import socket
import pickle
import time
import psutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from common.model import CNN
from common.utils import compute_DF

class Client:
    def __init__(self, client_id, edge_host, edge_port, rounds=5):
        self.id = client_id
        self.edge_host = edge_host
        self.edge_port = edge_port
        self.rounds = rounds
        self.model = CNN()
        self.CPU = min(psutil.cpu_percent(interval=1) / 100, 0.15)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
        self.DS = [150, 100, 200, 120, 180, 90, 160, 130, 140, 110][client_id]
        indices = list(range(self.DS * client_id, self.DS * (client_id + 1)))
        self.data = torch.utils.data.Subset(dataset, indices[:self.DS])
        self.DF = compute_DF(self.CPU, self.DS)

    def quantize_weights(self, state_dict):
        for key in state_dict:
            state_dict[key] = torch.round(state_dict[key] * 1000) / 1000
        return state_dict

    def train(self, bandwidth):
        loader = DataLoader(self.data, batch_size=10, shuffle=True)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        self.model.train()
        for _ in range(3):
            for data, target in loader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        training_time = time.time() - start_time
        self.E_c = training_time / bandwidth
        self.L_local = loss.item()

    def send_update(self):
        state_dict = self.quantize_weights(self.model.state_dict())
        state = [self.DF, self.DS, self.E_c, self.CPU, self.L_local]
        data = pickle.dumps({'weights': state_dict, 'DS': self.DS, 'state': state})
        
        print(f"Client {self.id} attempting to connect to {self.edge_host}:{self.edge_port}")
        for attempt in range(5):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(10)
                    s.connect((self.edge_host, self.edge_port))
                    s.sendall(data)
                print(f"Client {self.id} sent update successfully")
                return True
            except Exception as e:
                print(f"Client {self.id} failed to send update (attempt {attempt + 1}/5): {e}")
                time.sleep(3)  # Increased from 5s to 3s for faster retries
        print(f"Client {self.id} gave up sending update after 5 attempts")
        return False

    def run(self, bandwidth=1.0):
        for round_num in range(self.rounds):
            print(f"Client {self.id} starting Round {round_num + 1}/{self.rounds}")
            self.train(bandwidth)
            self.send_update()
            print(f"Client {self.id} - State: {self.DF, self.DS, self.E_c, self.CPU, self.L_local}")

if __name__ == "__main__":
    import sys
    client_id = int(sys.argv[1])
    edge_map = {0: 'edge0-1:5000', 1: 'edge0-1:5000', 2: 'edge0-1:5000',
                3: 'edge1-1:5001', 4: 'edge1-1:5001', 5: 'edge1-1:5001',
                6: 'edge2-1:5002', 7: 'edge2-1:5002',
                8: 'edge3-1:5003', 9: 'edge3-1:5003'}
    edge_host, edge_port = edge_map[client_id].split(':')
    client = Client(client_id, edge_host, int(edge_port), rounds=5)
    time.sleep(2)
    client.run(bandwidth=1.0)