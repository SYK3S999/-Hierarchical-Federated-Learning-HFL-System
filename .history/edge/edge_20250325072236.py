import socket
import pickle
import torch
from common.model import CNN
from common.utils import aggregate_models
import socketserver

class Edge:
    def __init__(self, edge_id, cloud_host, cloud_port, num_clients=2):
        self.id = edge_id
        self.cloud_host = cloud_host
        self.cloud_port = cloud_port
        self.num_clients = num_clients
        self.model = CNN()
        self.port = 5000 + edge_id
        self.DF_j = 0.0
        self.DS_j = 0.0
        self.E_c = 0.0
        self.CPU_j = 0.3
        self.L_avg = 0.0

    def listen_and_aggregate(self):
        client_updates = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen(self.num_clients)
            s.settimeout(30)  # Wait up to 30 seconds for clients
            try:
                for _ in range(self.num_clients):
                    conn, _ = s.accept()
                    data = b''
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    client_updates.append(pickle.loads(data))
                    conn.close()
            except socket.timeout:
                print(f"Edge {self.id} timed out waiting for {self.num_clients} clients, proceeding with {len(client_updates)} updates")
        
        if not client_updates:
            print(f"Edge {self.id} received no updates, skipping aggregation")
            return
        
        models = [CNN() for _ in client_updates]
        for model, update in zip(models, client_updates):
            model.load_state_dict(update['weights'])
        
        weights = [update['DS'] for update in client_updates]
        aggregate_models(self.model, models, weights)
        
        self.DF_j = sum(update['state'][0] for update in client_updates) / len(client_updates)
        self.DS_j = sum(weights)
        self.E_c = sum(update['state'][2] for update in client_updates)
        self.L_avg = sum(update['state'][4] for update in client_updates) / len(client_updates)
        
        state_dict = self.model.state_dict()
        state = [self.DF_j, self.DS_j, self.E_c, self.CPU_j, self.L_avg]
        data = pickle.dumps({'weights': state_dict, 'DS': self.DS_j, 'state': state})
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.cloud_host, self.cloud_port))
            s.sendall(data)

if __name__ == "__main__":
    import sys
    edge_id = int(sys.argv[1])
    num_clients = 3 if edge_id in [0, 3] else 2
    edge = Edge(edge_id, 'cloud', 6000, num_clients)
    edge.listen_and_aggregate()
    print(f"Edge {edge_id} - State: {edge.DF_j, edge.DS_j, edge.E_c, edge.CPU_j, edge.L_avg}")