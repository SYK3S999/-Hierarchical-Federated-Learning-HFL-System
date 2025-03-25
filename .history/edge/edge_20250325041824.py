import socket
import pickle
import torch
import random
import time
from common.model import CNN
from common.utils import aggregate_models

class Edge:
    def __init__(self, edge_id, cloud_host, cloud_port, client_count):
        self.id = edge_id
        self.cloud_host = cloud_host
        self.cloud_port = cloud_port
        self.client_count = client_count
        self.model = CNN()
        self.port = 5000 + edge_id
        self.DF_j = 0.0
        self.DS_j = 0.0
        self.E_c = 0.0
        self.CPU_j = 0.3
        self.L_avg = 0.0
        self.peers = [5000 + i for i in range(4) if i != edge_id]

    def share_with_peers(self, data):
        for peer_port in self.peers:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('0.0.0.0', peer_port))
                    s.sendall(data)
            except:
                continue

    def listen_and_aggregate(self):
        if random.random() < 0.1:
            print(f"Edge {self.id} failed this round")
            return False
        
        client_updates = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen(self.client_count)
            s.settimeout(10)
            try:
                for _ in range(self.client_count):
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
                print(f"Edge {self.id} timed out waiting for clients")
        
        if not client_updates:
            print(f"Edge {self.id} received no client updates")
            return False
        
        models = [CNN() for _ in client_updates]
        for model, update in zip(models, client_updates):
            model.load_state_dict({k: v.to(torch.float32) for k, v in update['weights'].items()})
        
        weights = [update['DS'] for update in client_updates]
        aggregate_models(self.model, models, weights)
        
        self.DF_j = sum(update['state'][0] for update in client_updates) / len(client_updates)
        self.DS_j = sum(weights)
        self.E_c = sum(update['state'][2] for update in client_updates)
        self.L_avg = sum(update['state'][4] for update in client_updates) / len(client_updates)
        
        state_dict = self.model.state_dict()
        state = [self.DF_j, self.DS_j, self.E_c, self.CPU_j, self.L_avg]
        data = pickle.dumps({'weights': state_dict, 'DS': self.DS_j, 'state': state})
        
        self.share_with_peers(data)
        
        # Retry mechanism
        for attempt in range(3):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5)
                    s.connect((self.cloud_host, self.cloud_port))
                    s.sendall(data)
                return True
            except Exception as e:
                print(f"Edge {self.id} failed to send to cloud (attempt {attempt + 1}/3): {e}")
                time.sleep(2)  # Wait before retrying
        print(f"Edge {self.id} gave up sending to cloud after 3 attempts")
        return False

if __name__ == "__main__":
    import sys
    edge_id = int(sys.argv[1])
    edge = Edge(edge_id, 'cloud', 6000, int(sys.argv[2]))
    edge.listen_and_aggregate()
    print(f"Edge {edge_id} - State: {edge.DF_j, edge.DS_j, edge.E_c, edge.CPU_j, edge.L_avg}")