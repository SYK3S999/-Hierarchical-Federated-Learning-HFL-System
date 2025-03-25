import socket
import pickle
import torch
from torch.utils.data import DataLoader
from common.model import CNN
from common.utils import aggregate_models
import time
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class Cloud:
    def __init__(self, rounds=5):
        logger.debug("Starting Cloud initialization")
        self.model = CNN()
        self.port = 6000
        self.rounds = rounds
        logger.debug("Loading test dataset")
        try:
            with open('/data/mnist_test_1000.pkl', 'rb') as f:
                self.testset = pickle.load(f)
            logger.debug("Test dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load test dataset: {e}")
            raise
        self.testloader = DataLoader(self.testset, batch_size=10, shuffle=False)
        logger.info("Cloud initialized successfully")

    def evaluate(self):
        logger.debug("Starting model evaluation")
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.testloader:
                output = self.model(data)
                _, pred = torch.max(output, 1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        acc = correct / total
        logger.debug(f"Evaluation complete: accuracy = {acc}")
        return acc

    def listen_and_aggregate(self):
        edge_updates = []
        edge_count = 4
        logger.debug(f"Attempting to bind socket to 0.0.0.0:{self.port}")
        for attempt in range(3):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(('0.0.0.0', self.port))
                s.listen(edge_count)
                s.settimeout(60)
                logger.info(f"Cloud listening on 0.0.0.0:{self.port} (attempt {attempt + 1})")
                break
            except Exception as e:
                logger.error(f"Failed to bind socket (attempt {attempt + 1}/3): {e}")
                time.sleep(2)
        else:
            logger.error("Cloud failed to bind after 3 attempts")
            return None, 0, 0, 0

        try:
            for _ in range(edge_count):
                conn, addr = s.accept()
                logger.info(f"Cloud received connection from {addr}")
                data = b''
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                edge_updates.append(pickle.loads(data))
                conn.close()
        except socket.timeout:
            logger.warning(f"Cloud timed out waiting for {edge_count - len(edge_updates)} edges")
        
        if not edge_updates:
            logger.warning("Cloud received no edge updates")
            return None, 0, 0, 0
        
        active_edges = len(edge_updates)
        if active_edges < edge_count:
            logger.info(f"Only {active_edges}/{edge_count} edges responded")
        
        models = [CNN() for _ in edge_updates]
        for model, update in zip(models, edge_updates):
            model.load_state_dict(update['weights'])
        
        weights = [update['DS'] for update in edge_updates]
        total_ds = sum(weights)
        weights = [w / total_ds if total_ds > 0 else 1/len(weights) for w in weights]
        
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
        logger.info(f"Starting cloud with {self.rounds} rounds")
        for round_num in range(self.rounds):
            logger.info(f"Starting Round {round_num + 1}/{self.rounds}")
            result = self.listen_and_aggregate()
            if result[0] is None:
                logger.warning(f"Round {round_num + 1} failed due to no edge updates, continuing to next round")
                continue  # Keep going even if a round fails
            state, reward, old_acc, acc = result
            logger.info(f"Round {round_num + 1} - State: {state}, Old Accuracy: {old_acc}, New Accuracy: {acc}, Reward: {reward}")
        logger.info("All rounds completed")

if __name__ == "__main__":
    logger.debug("Cloud script started")
    try:
        cloud = Cloud(rounds=5)
        cloud.run()
    except Exception as e:
        logger.error(f"Cloud crashed: {e}")
        raise
    logger.debug("Cloud script completed")