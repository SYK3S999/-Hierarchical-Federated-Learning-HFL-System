import torch
import socket
import pickle
import logging
import time
from common.model import CNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Edge:
    def __init__(self, edge_id, port, cloud_host='cloud-1', cloud_port=6000, rounds=5, num_clients=3):
        self.edge_id = edge_id
        self.port = port
        self.cloud_host = cloud_host
        self.cloud_port = cloud_port
        self.rounds = rounds
        self.num_clients = num_clients
        self.model = CNN()
        self.CPU = 0.3

    def listen_and_train(self):
        updates = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', self.port))
            s.listen(self.num_clients)
            s.settimeout(90)  # Bump to 90s
            logging.info(f"Edge {self.edge_id} listening on 0.0.0.0:{self.port}")
            start_time = time.time()
            while len(updates) < self.num_clients and (time.time() - start_time) < 90:
                try:
                    conn, addr = s.accept()
                    logging.info(f"Edge {self.edge_id} received connection from {addr}")
                    data = b""
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    updates.append(pickle.loads(data))
                    conn.close()
                except socket.timeout:
                    logging.warning(f"Edge {self.edge_id} timed out waiting for clients")
                    break
        
        if not updates:
            logging.info(f"Edge {self.edge_id} failed this round")
            return [0.0, 0, 0.0, self.CPU, 0.0], self.model.state_dict()
        
        weights = [update['state'][1] for update in updates]
        aggregated_weights = {}
        for key in updates[0]['weights'].keys():
            aggregated_weights[key] = torch.zeros_like(updates[0]['weights'][key])
            for update, weight in zip(updates, weights):
                aggregated_weights[key] += update['weights'][key] * weight
            aggregated_weights[key] /= sum(weights) + 1e-6
        
        self.model.load_state_dict(aggregated_weights)
        DF_avg = sum(u['state'][0] for u in updates) / len(updates)
        DS_total = sum(u['state'][1] for u in updates)
        E_c_avg = sum(u['state'][2] for u in updates) / len(updates)
        L_local_avg = sum(u['state'][4] for u in updates) / len(updates)
        state = [DF_avg, DS_total, E_c_avg, self.CPU, L_local_avg]
        return state, self.model.state_dict()

    def send_update_to_cloud(self, state, weights):
        data = pickle.dumps({'weights': weights, 'state': state})
        time.sleep(2)
        for attempt in range(10):  # More retries
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(10)
                    logging.info(f"Edge {self.edge_id} attempting to connect to {self.cloud_host}:{self.cloud_port}")
                    s.connect((self.cloud_host, self.cloud_port))
                    s.sendall(data)
                logging.info(f"Edge {self.edge_id} sent update to cloud successfully")
                return True
            except Exception as e:
                logging.error(f"Edge {self.edge_id} failed to send update (attempt {attempt + 1}/10): {e}")
                time.sleep(5)  # Longer retry delay
        logging.error(f"Edge {self.edge_id} gave up sending update after 10 attempts")
        return False

    def run(self):
        logging.info(f"Starting edge {self.edge_id} with {self.rounds} rounds")
        for round_num in range(self.rounds):
            logging.info(f"Edge {self.edge_id} starting Round {round_num + 1}/{self.rounds}")
            state, weights = self.listen_and_train()
            if self.send_update_to_cloud(state, weights):
                logging.info(f"Edge {self.edge_id} - State: {state}")
            else:
                logging.warning(f"Edge {self.edge_id} failed to send update to cloud")
            time.sleep(10)  # Longer sync delay
        logging.info(f"Edge {self.edge_id} completed rounds, waiting for cloud sync")
        time.sleep(30)

if __name__ == "__main__":
    import sys
    edge_id = int(sys.argv[1])
    ports = [5000, 5001, 5002, 5003]
    clients = [3, 3, 2, 2]
    edge = Edge(edge_id, ports[edge_id], num_clients=clients[edge_id])
    edge.run()