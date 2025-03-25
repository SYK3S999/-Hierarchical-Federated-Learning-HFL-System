import torch
import torch.nn as nn
import torch.optim as optim
import socket
import pickle
import logging
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from common.model import CNN
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Cloud:
    def __init__(self, rounds=5):
        self.model = CNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)  # Bump LR
        self.criterion = nn.CrossEntropyLoss()
        self.rounds = rounds
        self.CPU = min(psutil.cpu_percent(interval=1) / 100, 0.5)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        logging.info("Cloud initialized successfully")

    def receive_edge_updates(self, server_socket):
        updates = []
        server_socket.settimeout(60)
        start_time = time.time()
        while len(updates) < 4 and (time.time() - start_time) < 60:
            try:
                conn, addr = server_socket.accept()
                logging.info(f"Cloud received connection from {addr}")
                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                updates.append(pickle.loads(data))
                conn.close()
            except socket.timeout:
                logging.warning(f"Cloud timed out waiting for {4 - len(updates)} more edges")
                break
        return updates

    def listen_and_aggregate(self, server_socket):
        edge_updates = self.receive_edge_updates(server_socket)
        if not edge_updates:
            logging.warning("Cloud received no edge updates")
            return None
        
        valid_updates = [u for u in edge_updates if u['state'][1] > 0]
        if not valid_updates:
            logging.warning("No valid edge updates (all DS_total = 0)")
            return None
        
        weights = [u['state'][1] for u in valid_updates]
        aggregated_weights = {}
        for key in valid_updates[0]['weights'].keys():
            aggregated_weights[key] = torch.zeros_like(valid_updates[0]['weights'][key])
            for update, weight in zip(valid_updates, weights):
                aggregated_weights[key] += update['weights'][key] * weight
            aggregated_weights[key] /= sum(weights) + 1e-6
        
        self.model.load_state_dict(aggregated_weights)
        self.model.train()
        self.optimizer.zero_grad()
        for _ in range(5):  # Train on 5 batches
            for data, target in self.train_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                logging.debug(f"Training loss: {loss.item()}")
                break
        
        DF_avg = sum(u['state'][0] for u in valid_updates) / len(valid_updates)
        DS_total = sum(u['state'][1] for u in valid_updates)
        E_c_avg = sum(u['state'][2] for u in valid_updates) / len(valid_updates)
        L_local_avg = sum(u['state'][4] for u in valid_updates) / len(valid_updates)
        state = [DF_avg, DS_total, E_c_avg, self.CPU, L_local_avg]
        return state

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = correct / total
        logging.debug(f"Evaluation complete: accuracy = {accuracy}")
        return accuracy

    def compute_reward(self, old_acc, new_acc):
        return new_acc - old_acc - 0.5

    def run(self):
        logging.info(f"Starting cloud with {self.rounds} rounds")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', 6000))
            server_socket.listen(4)
            logging.info("Cloud listening on 0.0.0.0:6000")
            for round_num in range(self.rounds):
                logging.info(f"Starting Round {round_num + 1}/{self.rounds}")
                result = self.listen_and_aggregate(server_socket)
                if result:
                    old_acc = self.evaluate()
                    new_acc = self.evaluate()
                    reward = self.compute_reward(old_acc, new_acc)
                    logging.info(f"Round {round_num + 1} - State: {result}, Old Acc: {old_acc}, New Acc: {new_acc}, Reward: {reward}")
                else:
                    logging.warning(f"Round {round_num + 1} skipped due to no updates")
                time.sleep(5)
        logging.info("All rounds completed")

if __name__ == "__main__":
    cloud = Cloud(rounds=5)
    cloud.run()