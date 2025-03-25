import socket
import pickle
import torch
from common.model import CNN
from common.utils import aggregate_models

class Edge:
    def __init__(self, edge_id, cloud_host, cloud_port):
        self.id = edge_id
        self.cloud_host = cloud_host
        self.cloud_port = cloud_port
        self.model = CNN()
        self.port = 5000 + edge_id
        self.DF_j = 0.0
        self.DS_j = 0.0
        self.E_c = 0.0
        self.CPU_j = 0.3
        self.B_edge = 0.0

    def listen_and_aggregate(self):
        client_updates = []
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
                client_updates.append(pickle.loads(data))
                conn.close()
        
        models = [CNN() for _ in client_updates]
        for model, update in zip(models, client_updates):
            model.load_state_dict(update['weights'])
        
        weights = [update['DS'] for update in client_updates]
        aggregate_models(self.model, models, weights)
        
        self.DF_j = sum(update['state'][0] for update in client_updates) / len(client_updates)
        self.DS_j = sum(weights)
        self.E_c = sum(update['state'][2] for update in client_updates)
        # Fix: Compute client L_avg for this edge
        self.L_avg = sum(update['state'][4] for update in client_updates) / len(client_updates)
        
        state_dict = self.model.state_dict()
        state = [self.DF_j, self.DS_j, self.E_c, self.CPU_j, self.L_avg]  # Replace B_edge with L_avg
        data = pickle.dumps({'weights': state_dict, 'DS': self.DS_j, 'state': state})
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.cloud_host, self.cloud_port))
            s.sendall(data)

if __name__ == "__main__":
    import sys
    edge_id = int(sys.argv[1])
    edge = Edge(edge_id, 'cloud', 6000)
    edge.B_edge = float(sys.argv[2])
    edge.listen_and_aggregate()
    print(f"Edge {edge_id} - State: {edge.DF_j, edge.DS_j, edge.E_c, edge.CPU_j, edge.L_avg}")