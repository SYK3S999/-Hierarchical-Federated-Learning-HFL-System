import socket
import pickle
import torch
from torch.utils.data import DataLoader
from common.model import CNN
from common.utils import aggregate_models

class Cloud:
    def __init__(self, rounds=5):
        self.model = CNN()
        self.port = 6000
        self.rounds = rounds
        with open('/data/mnist_test_1000.pkl', 'rb') as f:
            self.testset = pickle.load(f)
        self.testloader = DataLoader(self.testset, batch_size=10, shuffle=False)

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
        edge_count = 4
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen(edge_count)
            for _ in range(edge_count):
                conn, _ = s.accept()
                data = b''
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                edge_updates.append(pickle.loads(data))
                conn.close()
        
        # Collect peer-shared updates if any edge failed
        active_edges = len(edge_updates)
        if active_edges < edge_count:
            print(f"Only {active_edges} edges responded; collecting peer updates")
            # Simplified: Assume edges share all updates; in practice, query peers

        models = [CNN() for _ in edge_updates]
        for model, update in zip(models, edge_updates):
            model.load_state_dict(update['weights'])
        
        weights = [update['DS'] for update in edge_updates]
        total_ds = sum(weights)
        weights = [w / total_ds if total_ds > 0 else 1/len(weights) for w in weights]  # Adjust for failures
        
        old_acc = self.evaluate()
        aggregate_models(self.model, models, weights)
        acc = self.evaluate()
        
        DF_avg = sum(update['state'][0] for update in edge_updates) / len(edge_updates)
        DS_total = sum(update['DS'] for update in edge_updates)
        E_c = sum(update['state'][2] for update in edge_updates) * 0.1
        E_total = sum(update['state'][2] for update in edge_updates) + E_c
        L_avg = sum(update['state'][4] for update in edge_updates) / len(edge_updates)
        reward = 1.0 * (acc - old_acc) - 0.5 * E_total - 0.5 * L_avg
        
        return [DF_avg, DS_total, E_c, 1.0, 0.5], reward, old_acc, acc

    def run(self):
        for round_num in range(self.rounds):
            print(f"Starting Round {round_num + 1}/{self.rounds}")
            state, reward, old_acc, acc = self.listen_and_aggregate()
            print(f"Round {round_num + 1} - State: {state}, Old Accuracy: {old_acc}, New Accuracy: {acc}, Reward: {reward}")
            # Persist model state for next round (simplified: in-memory here)

if __name__ == "__main__":
    cloud = Cloud(rounds=5)
    cloud.run()