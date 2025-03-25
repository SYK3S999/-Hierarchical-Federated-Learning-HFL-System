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
        with open('/data/mnist_train_variable.pkl', 'rb') as f:
            self.data = pickle.load(f)[client_id]
        self.model = CNN()
        self.DS = len(self.data)
        self.CPU = min(psutil.cpu_percent(interval=1) / 100, 0.15)

    def quantize_weights(self, state_dict):
        return {k: v.to(torch.float16) for k, v in state_dict.items()}

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
        training_time = time.time() - start_time
        
        self.E_c = training_time
        self.L_local = (self.DS * 0.001) / (bandwidth + 1e-6) + training_time
        self.CPU = min(psutil.cpu_percent(interval=1) / 100, 0.15)
        
        v_c = torch.var(torch.tensor([t for _, t in self.data], dtype=torch.float32))
        self.DF = compute_DF(v_c, self.DS).item()

    def send_update(self):
        state_dict = self.quantize_weights(self.model.state_dict())
        state = [self.DF, self.DS, self.E_c, self.CPU, self.L_local]
        data = pickle.dumps({'weights': state_dict, 'DS': self.DS, 'state': state})
        
        print(f"Client {self.id} attempting to connect to {self.edge_host}:{self.edge_port}")
        for attempt in range(3):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5)
                    s.connect((self.edge_host, self.edge_port))
                    s.sendall(data)
                print(f"Client {self.id} sent update successfully")
                return True
            except Exception as e:
                print(f"Client {self.id} failed to send update (attempt {attempt + 1}/3): {e}")
                time.sleep(1)
        print(f"Client {self.id} gave up sending update after 3 attempts")
        return False

if __name__ == "__main__":
    client_id = int(sys.argv[1])
    edge_id = 0 if client_id < 3 else (1 if client_id < 6 else (2 if client_id < 8 else 3))
    edge_host = f'edge{edge_id}-1'  # Match container name
    edge_port = 5000 + edge_id
    
    time.sleep(2)  # Wait for edge to start
    client = Client(client_id, edge_host, edge_port)
    client.train(float(sys.argv[2]))
    client.send_update()
    print(f"Client {client_id} - State: {client.DF, client.DS, client.E_c, client.CPU, client.L_local}")