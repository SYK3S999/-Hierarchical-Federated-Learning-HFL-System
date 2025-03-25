import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import socket
import psutil
import time
import sys
from common.model import CNN
from common.utils import compute_DF

class Client:
    def __init__(self, client_id, edge_host, edge_port):
        self.id = client_id
        self.edge_host = edge_host
        self.edge_port = edge_port
        with open('/data/mnist_train_400.pkl', 'rb') as f:
            self.data = pickle.load(f)[client_id]
        self.model = CNN()
        self.DS = len(self.data)
        self.CPU = psutil.cpu_percent(interval=1) / 100

    def train(self, bandwidth):
        loader = DataLoader(self.data, batch_size=10, shuffle=True)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        self.model.train()
        for data, target in loader:
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        elapsed = time.time() - start_time
        self.E_c = 0.01 * self.DS / self.CPU
        self.L_local = 0.1 * self.DS / (bandwidth * self.CPU + 1e-6)
        self.CPU = psutil.cpu_percent(interval=1) / 100
        
        v_c = torch.var(torch.tensor([t.item() for _, t in self.data], dtype=torch.float32))
        self.DF = compute_DF(v_c, self.DS).item()

    def send_update(self):
        state_dict = self.model.state_dict()
        state = [self.DF, self.DS, self.E_c, self.CPU, self.L_local]
        data = pickle.dumps({'weights': state_dict, 'DS': self.DS, 'state': state})
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.edge_host, self.edge_port))
            s.sendall(data)

if __name__ == "__main__":
    client_id = int(sys.argv[1])
    edge_host = 'edge0' if client_id < 2 else 'edge1'
    edge_port = 5000 + (client_id // 2)
    bandwidth = float(sys.argv[2])
    
    client = Client(client_id, edge_host, edge_port)
    client.train(bandwidth)
    client.send_update()
    print(f"Client {client_id} - State: {client.DF, client.DS, client.E_c, client.CPU, client.L_local}")