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
        
        models = [CNN() for _ in edge_updates]
        for model, update in zip(models, edge_updates):
            model.load_state_dict(update['weights'])
        
        weights = [update['DS'] for update in edge_updates]
        old_acc = self.evaluate()
        aggregate_models(self.model, models, weights)
        acc = self.evaluate()
        
        self.DF_avg = sum(update['state'][0] for update in edge_updates) / len(edge_updates)
        self.DS_total = sum(weights)
        self.E_c = sum(update['state'][2] for update in edge_updates) * 0.1
        
        # Fix: Correctly compute E_total and L_avg from client data
        client_E_total = sum(update['state'][2] for update in edge_updates)  # Sum of client E_c from edges
        E_total = client_E_total + self.E_c
        client_L_values = [update['state'][4] for update in edge_updates]  # Assumes edge passes L_avg
        L_avg = sum(client_L_values) / len(client_L_values)
        reward = 1.0 * (acc - old_acc) - 0.5 * E_total - 0.5 * L_avg
        
        state_dict = {
            "DF_avg (Avg Data Heterogeneity)": self.DF_avg,
            "DS_total (Total Dataset Size)": self.DS_total,
            "E_c (Cloud Energy Cost)": self.E_c,
            "B_available (Available Bandwidth)": self.B_available,
            "CPU_cloud (Cloud CPU Utilization)": self.CPU_cloud
        }
        return {'s_g': state_dict}, reward, old_acc, acc

if __name__ == "__main__":
    cloud = Cloud()
    state, reward, old_acc, acc = cloud.listen_and_aggregate()
    # Print state with names, plus additional metrics
    print(f"Cloud - State: {state['s_g']}, Old Accuracy: {old_acc:.4f}, New Accuracy: {acc:.4f}, Reward: {reward:.4f}")