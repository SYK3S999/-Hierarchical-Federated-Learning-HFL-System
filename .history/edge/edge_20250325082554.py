import socket
import pickle
import torch
from common.model import CNN
from common.utils import aggregate_models, setup_logging, hash_weights

class Edge:
    def __init__(self, edge_id, cloud_host, cloud_port):
        self.logger = setup_logging(f"Edge_{edge_id}")
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
        self.weight_hash_avg = 0
        self.logger.info(f"Initialized with port {self.port}, CPU: {self.CPU_j}")

    def listen_and_aggregate(self):
        self.logger.info("Starting to listen for client updates")
        client_updates = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen(2)
            self.logger.info(f"Listening on port {self.port}")
            for i in range(2):
                conn, addr = s.accept()
                self.logger.info(f"Accepted connection from {addr}")
                data = b''
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                client_updates.append(pickle.loads(data))
                conn.close()
                self.logger.info(f"Received update from client {i + 1}/2")
        
        self.logger.info("Aggregating client models")
        models = [CNN() for _ in client_updates]
        for model, update in zip(models, client_updates):
            model.load_state_dict(update['weights'])
        
        weights = [update['DS'] for update in client_updates]
        aggregate_models(self.model, models, weights)
        
        self.DF_j = sum(update['state'][0] for update in client_updates) / len(client_updates)
        self.DS_j = sum(weights)
        self.E_c = sum(update['state'][2] for update in client_updates)
        self.L_avg = sum(update['state'][4] for update in client_updates) / len(client_updates)
        self.weight_hash_avg = sum(update['state'][5] for update in client_updates) / len(client_updates)
        
        self.logger.info(f"Aggregation complete: DF_j: {self.DF_j:.3f}, DS_j: {self.DS_j}, E_c: {self.E_c:.3f}, L_avg: {self.L_avg:.3f}, Weight Hash Avg: {self.weight_hash_avg}")
        
        state_dict = self.model.state_dict()
        state = [self.DF_j, self.DS_j, self.E_c, self.CPU_j, self.L_avg, self.weight_hash_avg]  # 6 elements
        data = pickle.dumps({'weights': state_dict, 'DS': self.DS_j, 'state': state})
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            self.logger.info(f"Connecting to cloud at {self.cloud_host}:{self.cloud_port}")
            s.connect((self.cloud_host, self.cloud_port))
            self.logger.info("Connected, sending aggregated update")
            s.sendall(data)
            s.shutdown(socket.SHUT_WR)
            self.logger.info("Update sent to cloud")

if __name__ == "__main__":
    import sys
    edge_id = int(sys.argv[1])
    edge = Edge(edge_id, 'cloud', 6000)
    edge.B_edge = float(sys.argv[2])
    edge.listen_and_aggregate()
    
    state_dict = {
        "DF_j (Avg Data Heterogeneity)": edge.DF_j,
        "DS_j (Total Dataset Size)": edge.DS_j,
        "E_c (Total Energy Cost)": edge.E_c,
        "CPU_j (Edge CPU Utilization)": edge.CPU_j,
        "L_avg (Avg Latency)": edge.L_avg,
        "Weight Hash Avg": edge.weight_hash_avg
    }
    edge.logger.info(f"Final state: {state_dict}")