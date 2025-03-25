import socket
import pickle
import torch
from common.model import CNN
from common.utils import aggregate_models

class Cloud:
    def __init__(self, num_edges=4):
        self.model = CNN()
        self.port = 6000
        self.num_edges = num_edges

    def listen_and_aggregate(self):
        edge_updates = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen(self.num_edges)
            for _ in range(self.num_edges):
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
        aggregate_models(self.model, models, weights)
        
        # Aggregate metrics
        DF_global = sum(update['state'][0] for update in edge_updates) / len(edge_updates)
        DS_global = sum(weights)
        E_c_global = sum(update['state'][2] for update in edge_updates)
        L_avg_global = sum(update['state'][4] for update in edge_updates) / len(edge_updates)
        
        print(f"Cloud - Global State: {DF_global, DS_global, E_c_global, L_avg_global}")

if __name__ == "__main__":
    cloud = Cloud(num_edges=4)
    cloud.listen_and_aggregate()