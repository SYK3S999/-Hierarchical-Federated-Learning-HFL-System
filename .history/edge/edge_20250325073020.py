# edge.py
import socket
import pickle
import torch
from common.model import CNN
from common.utils import aggregate_models
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.info(f"Edge {self.id} initialized, expecting {self.num_clients} clients")

    def listen_and_aggregate(self):
        client_updates = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen(self.num_clients)
            s.settimeout(30)
            try:
                for i in range(self.num_clients):
                    conn, addr = s.accept()
                    logging.info(f"Edge {self.id} accepted connection from client {i} at {addr}")
                    try:
                        # Receive the 4-byte length prefix
                        length_bytes = conn.recv(4)
                        if not length_bytes:
                            logging.warning(f"Edge {self.id} received no length prefix from client {i} at {addr}")
                            conn.close()
                            continue
                        data_len = int.from_bytes(length_bytes, byteorder='big')
                        data = b''
                        while len(data) < data_len:
                            chunk = conn.recv(min(4096, data_len - len(data)))
                            if not chunk:
                                break
                            data += chunk
                        conn.close()
                        if len(data) == data_len:
                            client_updates.append(pickle.loads(data))
                            logging.info(f"Edge {self.id} received update from client {i} at {addr}, size={len(data)} bytes")
                        else:
                            logging.warning(f"Edge {self.id} received incomplete data from client {i} at {addr}, got {len(data)}/{data_len} bytes")
                    except Exception as e:
                        logging.error(f"Edge {self.id} error receiving data from client {i} at {addr}: {e}")
                        conn.close()
            except socket.timeout:
                logging.warning(f"Edge {self.id} timed out, received {len(client_updates)}/{self.num_clients} updates")
        
        if not client_updates:
            logging.error(f"Edge {self.id} received no valid updates, skipping aggregation")
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
        logging.info(f"Edge {self.id} aggregated {len(client_updates)} client updates")
        
        state_dict = self.model.state_dict()
        state = [self.DF_j, self.DS_j, self.E_c, self.CPU_j, self.L_avg]
        data = pickle.dumps({'weights': state_dict, 'DS': self.DS_j, 'state': state})
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.cloud_host, self.cloud_port))
                s.sendall(len(data).to_bytes(4, byteorder='big'))
                s.sendall(data)
                logging.info(f"Edge {self.id} sent update to cloud, size={len(data)} bytes")
        except Exception as e:
            logging.error(f"Edge {self.id} failed to send update to cloud: {e}")

if __name__ == "__main__":
    import sys
    edge_id = int(sys.argv[1])
    num_clients = 3 if edge_id in [0, 3] else 2
    edge = Edge(edge_id, 'cloud', 6000, num_clients)
    edge.listen_and_aggregate()
    print(f"Edge {edge_id} - State: {edge.DF_j, edge.DS_j, edge.E_c, edge.CPU_j, edge.L_avg}")