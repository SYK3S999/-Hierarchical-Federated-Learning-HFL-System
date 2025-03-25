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
        logging.debug("Cloud script started")
        logging.debug("Starting Cloud initialization")
        self.model = CNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.rounds = rounds
        self.CPU = min(psutil.cpu_percent(interval=1) / 100, 0.5)
        logging.debug("Loading test dataset")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        logging.debug("Test dataset loaded successfully")
        logging.info("Cloud initialized successfully")

    def receive_edge_updates(self):
        updates = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            logging.debug("Attempting to bind socket to 0.0.0.0:6000")
            s.bind(('0.0.0.0', 6000))
            s.listen(4)
            s.settimeout(120)  # Increase to 120s
            logging.info("Cloud listening on 0.0.0.0:6000 (attempt 1)")
            for _ in range(4):  # Expect 4 edges
                try:
                    conn, addr = s.accept()
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

    def listen_and_aggregate(self):
        edge_updates = self.receive_edge_updates()
        if not edge_updates:
            logging.warning("Cloud received no edge updates")
            return None
        
        # Extract DS from state[1] and aggregate weights
        weights = [update['state'][1] for update in edge_updates]  # DS_total is state[1]
        aggregated_weights = {}
        for key in edge_updates[0]['weights'].keys():
            aggregated_weights[key] = torch.zeros_like(edge_updates[0]['weights'][key])
            for update, weight in zip(edge_updates, weights):
                aggregated_weights[key] += update['weights'][key] * weight
            aggregated_weights[key] /= sum(weights) + 1e-6
        
        self.model.load_state_dict(aggregated_weights)
        
        # Compute cloud state
        DF_avg = sum(u['state'][0] for u in edge_updates) / len(edge_updates)
        DS_total = sum(u['state'][1] for u in edge_updates)
        E_c_avg = sum(u['state'][2] for u in edge_updates) / len(edge_updates)
        L_local_avg = sum(u['state'][4] for u in edge_updates) / len(edge_updates)
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
        return new_acc - old_acc - 0.5  # Simple reward function, adjust as needed

    def run(self):
        logging.info(f"Starting cloud with {self.rounds} rounds")
        for round_num in range(self.rounds):
            logging.info(f"Starting Round {round_num + 1}/{self.rounds}")
            result = self.listen_and_aggregate()
            if result:
                old_acc = self.evaluate()
                self.optimizer.step()
                new_acc = self.evaluate()
                reward = self.compute_reward(old_acc, new_acc)
                logging.info(f"Round {round_num + 1} - State: {result}, Old Accuracy: {old_acc}, New Accuracy: {new_acc}, Reward: {reward}")
            logging.debug("Round completed")
        logging.info("All rounds completed")
        logging.debug("Cloud script completed")

if __name__ == "__main__":
    cloud = Cloud(rounds=5)
    cloud.run()