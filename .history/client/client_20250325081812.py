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
from common.utils import compute_DF, setup_logging

class Client:
    def __init__(self, client_id, edge_host, edge_port):
        self.logger = setup_logging(f"Client_{client_id}")
        self.id = client_id
        self.edge_host = edge_host
        self.edge_port = edge_port
        self.logger.info("Initializing client")
        with open('/data/mnist_train_400.pkl', 'rb') as f:
            self.data = pickle.load(f)[client_id]
        self.model = CNN()
        self.DS = len(self.data)
        self.CPU = psutil.cpu_percent(interval=1) / 100
        self.logger.info(f"Loaded {self.DS} data samples, initial CPU: {self.CPU:.2f}")

    def train(self, bandwidth, epochs=2):
        self.logger.info(f"Starting training with {epochs} epochs, bandwidth: {bandwidth}")
        loader = DataLoader(self.data, batch_size=10, shuffle=True)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for data, target in loader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.logger.info(f"Epoch {epoch + 1}/{epochs} completed, avg loss: {epoch_loss / len(loader):.4f}")
        
        elapsed = time.time() - start_time
        self.E_c = 0.01 * self.DS / self.CPU
        self.L_local = 0.1 * self.DS / (bandwidth * self.CPU + 1e-6)
        self.CPU = psutil.cpu_percent(interval=1) / 100
        
        v_c = torch.var(torch.tensor([t for _, t in self.data], dtype=torch.float32))
        self.DF = compute_DF(v_c, self.DS).item()
        self.logger.info(f"Training finished in {elapsed:.2f}s, E_c: {self.E_c:.3f}, L_local: {self.L_local:.3f}, DF: {self.DF:.3f}")

    def send_update(self):
        self.logger.info("Preparing to send update to edge")
        state_dict = self.model.state_dict()
        state = [self.DF, self.DS, self.E_c, self.CPU, self.L_local]
        data = pickle.dumps({'weights': state_dict, 'DS': self.DS, 'state': state})
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            self.logger.info(f"Connecting to {self.edge_host}:{self.edge_port}")
            s.connect((self.edge_host, self.edge_port))
            self.logger.info("Connected, sending data")
            s.sendall(data)
            s.shutdown(socket.SHUT_WR)
            self.logger.info("Update sent successfully")

if __name__ == "__main__":
    client_id = int(sys.argv[1])
    edge_host = 'edge0' if client_id < 2 else 'edge1'
    edge_port = 5000 + (client_id // 2)
    bandwidth = float(sys.argv[2])
    
    client = Client(client_id, edge_host, edge_port)
    client.train(bandwidth, epochs=2)
    client.send_update()
    state_dict = {
        "DF (Data Quality factor)": client.DF,
        "DS (Dataset Size)": client.DS,
        "E_c (Energy Cost)": client.E_c,
        "CPU (Utilization)": client.CPU,
        "L_local (Latency)": client.L_local
    }
    client.logger.info(f"Final state: {state_dict}")