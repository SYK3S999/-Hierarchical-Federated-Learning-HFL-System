import torch
import socket
import pickle
import logging
import time
import random
import psutil
from common.model import CNN
from common.utils import aggregate_weights, compute_DF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Edge:
    def __init__(self, edge_id, cloud_host, cloud_port, rounds=5):
        self.edge_id = edge_id
        self.cloud_host = cloud_host
        self.cloud_port = cloud_port
        self.port = 5000 + edge_id
        self.model = CNN()
        self.rounds = rounds
        self.client_count = {0: 3, 1: 3, 2: 2, 3: 2}[edge_id]
        logging.debug(f"Edge {self.edge_id} initialization started")
        self.CPU = min(psutil.cpu_percent(interval=1) / 100, 0.3)
        logging.debug(f"Edge {self.edge_id} initialized")

    def receive_client_updates(self):
        updates = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            logging.debug(f"Edge {self.edge_id} attempting to bind to 0.0.0.0:{self.port}")
            s.bind(('0.0.0.0', self.port))
            s.listen(self.client_count)
            s.settimeout(30)
            logging.info(f"Edge {self.edge_id} listening on 0.0.0.0:{self.port}")
            
            for _ in range(self.client_count):
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
                    logging.warning(f"Edge {self.edge_id} timed out waiting for client")
                    break
        return updates

    def send_update_to_cloud(self, state, weights):
        data = pickle.dumps({'weights': weights, 'state': state})
        for attempt in range(5):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5)
                    logging.info(f"Edge {self.edge_id} attempting to connect to {self.cloud_host}:{self.cloud_port}")
                    s.connect((self.cloud_host, self.cloud_port))
                    s.sendall(data)
                logging.info(f"Edge {self.edge_id} sent update to cloud successfully")
                return True
            except Exception as e:
                logging.error(f"Edge {self.edge_id} failed to send update (attempt {attempt + 1}/5): {e}")
                time.sleep(2)
        logging.error(f"Edge {self.edge_id} gave up sending update after 5 attempts")
        return False

    def run(self):
        logging.info(f"Starting edge {self.edge_id} with {self.rounds} rounds")
        for round_num in range(self.rounds):
            logging.info(f"Edge {self.edge_id} starting Round {round_num + 1}/{self.rounds}")
            if random.random() < 0.1:
                logging.info(f"Edge {self.edge_id} failed this round")
                state = [0.0, 0.0, 0.0, self.CPU, 0.0]
                logging.info(f"Edge {self.edge_id} - State: {state}")
                continue

            updates = self.receive_client_updates()
            if not updates:
                logging.warning(f"Edge {self.edge_id} received no client updates")
                state = [0.0, 0.0, 0.0, self.CPU, 0.0]
            else:
                weights = aggregate_weights([u['weights'] for u in updates])
                self.model.load_state_dict(weights)
                DS_total = sum(u['DS'] for u in updates)
                DF_avg = sum(u['state'][0] for u in updates) / len(updates)
                E_c_avg = sum(u['state'][2] for u in updates) / len(updates)
                L_local_avg = sum(u['state'][4] for u in updates) / len(updates)
                state = [DF_avg, DS_total, E_c_avg, self.CPU, L_local_avg]
                self.send_update_to_cloud(state, weights)

            logging.info(f"Edge {self.edge_id} - State: {state}")
            time.sleep(1)  # Sync delay

        # Wait for cloud to finish all rounds
        logging.info(f"Edge {self.edge_id} completed rounds, waiting for cloud sync")
        time.sleep(60)  # Extra time to ensure cloud processes Round 5

if __name__ == "__main__":
    import sys
    edge_id = int(sys.argv[1])
    edge = Edge(edge_id, 'cloud-1', 6000, rounds=5)
    edge.run()
    logging.debug("Edge script completed")