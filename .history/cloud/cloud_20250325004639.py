import socket
import pickle
import torch
from torch.utils.data import DataLoader
from common.model import CNN
from common.utils import aggregate_models

class Cloud:
    def __init__(self):
        self.model = CNN()
        self.port = 6000
        with open('/data/mnist_test_100.pkl', 'rb') as f:
            self.testset = pickle.load(f)
        self.testloader = DataLoader(self.testset, batch_size=10, shuffle=False)
        self.DF_avg = 0.0
        self.DS_total = 0.0
        self.E_c = 0.0
        self.B_available = 1.0
        self.CPU_cloud = 0.5

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.testloader:
                output = self.model(data)
                _, pred = torch.max(output, 1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        return correct / total

    def listen_and_aggregate(self):
        edge_updates = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen(2)
            for _ in range(2):
                conn, _ = s.accept()
                data = b''
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                edge_updates.append(pickle.loads(data))
                conn.close()
        
        # Fix: Create CNN instances and load state dicts correctly
        models = [CNN() for _ in edge_updates]  # Create list of CNN instances
        for model, update in zip(models, edge_updates):
            model.load_state_dict(update['weights'])  # Load weights into each model
        
        weights = [update['DS'] for update in edge_updates]
        old_acc = self.evaluate()
        aggregate_models(self.model, models, weights)
        acc = self.evaluate()
        
        self.DF_avg = sum(update['state'][0] for update in edge_updates) / len(edge_updates)
        self.DS_total = sum(weights)
        self.E_c = sum(update['state'][2] for update in edge_updates) * 0.1
        
        E_total = sum(update['state'][2] for update in edge_updates) + self.E_c
        L_avg = sum(update['state'][4] for update in edge_updates) / len(edge_updates)
        reward = 1.0 * (acc - old_acc) - 0.5 * E_total - 0.5 * L_avg
        
        return {'s_g': [self.DF_avg, self.DS_total, self.E_c, self.B_available, self.CPU_cloud]}, reward

if __name__ == "__main__":
    cloud = Cloud()
    state, reward = cloud.listen_and_aggregate()
    print(f"Cloud - State: {state['s_g']}, Reward: {reward}")