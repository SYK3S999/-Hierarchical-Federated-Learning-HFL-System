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
    def __init__(self, client_id, edge_host, edge_port, epochs=1):
        self.id = client_id
        self.edge_host = edge_host
        self.edge_port = edge_port
        self.epochs = epochs
        with open('/data/mnist_train_10clients.pkl', 'rb') as f:
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
        for epoch in range(self.epochs):
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
        
        v_c = torch.var(torch.tensor([t for _, t in self.data], dtype=torch.float32))
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
    bandwidth = float(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    # Map clients to edges
    edge_mapping = {
        0: ('edge0', 5000), 1: ('edge0', 5000), 2: ('edge0', 5000),
        3: ('edge1', 5001), 4: ('edge1', 5001),
        5: ('edge2', 5002), 6: ('edge2', 5002),
        7: ('edge3', 5003), 8: ('edge3', 5003), 9: ('edge3', 5003)
    }
    edge_host, edge_port = edge_mapping[client_id]
    
    client = Client(client_id, edge_host, edge_port, epochs)
    client.train(bandwidth)
    client.send_update()
    print(f"Client {client_id} - State: {client.DF, client.DS, client.E_c, client.CPU, client.L_local}")