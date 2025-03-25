# cloud.py
import socket
import pickle
import torch
from common.model import CNN
from common.utils import aggregate_models
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Cloud:
    def __init__(self, num_edges=4):
        self.model = CNN()
        self.port = 6000
        self.num_edges = num_edges
        logging.info(f"Cloud initialized, expecting {self.num_edges} edges")

    def listen_and_aggregate(self):
        edge_updates = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen(self.num_edges)
            s.settimeout(60)
            try:
                for i in range(self.num_edges):
                    conn, addr = s.accept()
                    logging.info(f"Cloud accepted connection from edge {i} at {addr}")
                    try:
                        length_bytes = conn.recv(4)
                        if not length_bytes:
                            logging.warning(f"Cloud received no length prefix from edge {i} at {addr}")
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
                            edge_updates.append(pickle.loads(data))
                            logging.info(f"Cloud received update from edge {i} at {addr}, size={len(data)} bytes")
                        else:
                            logging.warning(f"Cloud received incomplete data from edge {i} at {addr}, got {len(data)}/{data_len} bytes")
                    except Exception as e:
                        logging.error(f"Cloud error receiving data from edge {i} at {addr}: {e}")
                        conn.close()
            except socket.timeout:
                logging.warning(f"Cloud timed out, received {len(edge_updates)}/{self.num_edges} updates")
        
        if not edge_updates:
            logging.error("Cloud received no valid updates, exiting")
            return
        
        models = [CNN() for _ in edge_updates]
        for model, update in zip(models, edge_updates):
            model.load_state_dict(update['weights'])
        
        weights = [update['DS'] for update in edge_updates]
        aggregate_models(self.model, models, weights)
        
        DF_global = sum(update['state'][0] for update in edge_updates) / len(edge_updates)
        DS_global = sum(weights)
        E_c_global = sum(update['state'][2] for update in edge_updates)
        L_avg_global = sum(update['state'][4] for update in edge_updates) / len(edge_updates)
        
        logging.info(f"Cloud aggregated {len(edge_updates)} edge updates")
        print(f"Cloud - Global State: {DF_global, DS_global, E_c_global, L_avg_global}")

if __name__ == "__main__":
    cloud = Cloud(num_edges=4)
    cloud.listen_and_aggregate()