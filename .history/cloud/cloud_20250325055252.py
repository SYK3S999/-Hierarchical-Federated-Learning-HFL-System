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
        edge_updates = self.receive_edge_updates()
        if not edge_updates:
            logging.warning("Cloud received no edge updates")
            return None
        
        # Extract DS from state[1] and aggregate weights
        weights = [update['state'][1] for update in edge_updates]  # DS_total is state[1]
        aggregated_weights = {}
        for key in edge_updates[0]['weights'].keys():
            aggregated_weights[key] = torch.zeros_like(edge_updates[0]['weights'][key])
            for update, weight in zip(edge_updates, weights):
                aggregated_weights[key] += update['weights'][key] * weight
            aggregated_weights[key] /= sum(weights) + 1e-6
        
        self.model.load_state_dict(aggregated_weights)
        
        # Compute cloud state
        DF_avg = sum(u['state'][0] for u in edge_updates) / len(edge_updates)
        DS_total = sum(u['state'][1] for u in edge_updates)
        E_c_avg = sum(u['state'][2] for u in edge_updates) / len(edge_updates)
        L_local_avg = sum(u['state'][4] for u in edge_updates) / len(edge_updates)
        state = [DF_avg, DS_total, E_c_avg, self.CPU, L_local_avg]
        
        return state

    def run(self):
        logging.info(f"Starting cloud with {self.rounds} rounds")
        for round_num in range(self.rounds):
            logging.info(f"Starting Round {round_num + 1}/{self.rounds}")
            result = self.listen_and_aggregate()
            if result:
                old_acc = self.evaluate()
                self.optimizer.step()
                new_acc = self.evaluate()
                reward = self.compute_reward(old_acc, new_acc)
                logging.info(f"Round {round_num + 1} - State: {result}, Old Accuracy: {old_acc}, New Accuracy: {new_acc}, Reward: {reward}")
            logging.debug("Round completed")
        logging.info("All rounds completed")

if __name__ == "__main__":
    logger.debug("Cloud script started")
    try:
        cloud = Cloud(rounds=5)
        cloud.run()
    except Exception as e:
        logger.error(f"Cloud crashed: {e}")
        raise
    logger.debug("Cloud script completed")