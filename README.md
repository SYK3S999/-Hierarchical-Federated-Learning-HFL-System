# Hierarchical Federated Learning (HFL) System

## Overview
This project implements a **Hierarchical Federated Learning (HFL)** system designed for distributed machine learning across multiple clients, edges, and a central cloud server. The system simulates a federated learning environment where clients train local models on private data, edges aggregate client updates, and the cloud aggregates edge updates to produce a global model. It includes a reward mechanism to evaluate system performance, serving as a baseline for future Hierarchical Reinforcement Learning (HRL) integration to optimize client selection and resource allocation.

The current implementation uses the MNIST dataset, a simple CNN model, and Docker containers to emulate a distributed setup. It supports non-IID data, variable computational resources, and a hierarchical communication structure.

---

## System Architecture
The HFL system is structured in three layers:
1. **Clients**: 4 nodes that train local models on subsets of MNIST data.
2. **Edges**: 2 nodes that aggregate updates from 2 clients each.
3. **Cloud**: 1 node that aggregates edge updates, evaluates the global model, and computes a reward.

### Components
- **Clients**: Train a CNN on local data, compute energy (\( E_c \)), latency (\( L_{\text{local}} \)), and data freshness (\( DF \)), then send updates to their edge.
- **Edges**: Aggregate client models and states, then forward to the cloud.
- **Cloud**: Aggregates edge models, evaluates accuracy on a test set, and calculates a reward based on accuracy improvement, energy, and latency.

### Key Metrics
- **State (\( s_c, s_e, s_g \))**: Includes \( DF \), dataset size (\( DS \)), \( E_c \), CPU usage, and \( L_{\text{local}} \) or bandwidth (\( B \)).
- **Reward**: \( R = 1.0 \cdot (acc - old_acc) - 0.5 \cdot E_{\text{total}} - 0.5 \cdot L_{\text{avg}} \), balancing accuracy gain against resource costs.

---

## Project Structure
  
``` bash
HFL_HRL/
├── client/
│   ├── Dockerfile          # Docker config for clients
│   ├── client.py           # Client training and update logic
│   └── requirements.txt    # Client dependencies (torch, psutil)
├── cloud/
│   ├── Dockerfile          # Docker config for cloud
│   ├── cloud.py            # Cloud aggregation and evaluation logic
│   └── requirements.txt    # Cloud dependencies (torch, torchvision)
├── common/
│   ├── model.py            # CNN model definition
│   ├── utils.py            # Utility functions (aggregation, DF computation)
│   └── init.py         # Makes common a Python package
├── data/
│   ├── prepare_data.py     # Script to generate MNIST subsets
│   ├── mnist_train_400.pkl # Training data (400 images, 100 per client)
│   └── mnist_test_100.pkl  # Test data (100 images)
├── edge/
│   ├── Dockerfile          # Docker config for edges
│   ├── edge.py             # Edge aggregation and communication logic
│   └── requirements.txt    # Edge dependencies (torch, numpy)
├── docker-compose.yml      # Orchestrates all containers
└── README.md               # This file

```

---

## Prerequisites
- **Docker**: Installed and running (tested with Docker Desktop on Windows).
- **Python**: Used in data preparation (3.9 recommended).
- **CUDA**: Optional for GPU acceleration (not currently utilized).

---

## Setup
1. **Prepare Data**:
   - Navigate to the `data/` directory:
     ```bash
     cd data
     python prepare_data.py
This generates mnist_train_400.pkl (4 client subsets, 100 images each) and mnist_test_100.pkl (100 test images).

2. **Build and Run**:
From the root directory (HFL_HRL/):
``` bash
docker-compose up --build
```
- Builds and starts all containers (4 clients, 2 edges, 1 cloud).
## Usage
### Run the System:
- Execute docker-compose up --build to start a single round of HFL.
- Containers will:
  - Train local models (clients).
  - Aggregate updates (edges).
  - Compute global model, accuracy, and reward (cloud).

## Current Features
- Non-IID Data: Clients train on distinct MNIST subsets (e.g., different digit distributions).
- Hierarchical Aggregation: Two levels of model averaging (client-to-edge, edge-to-cloud).
- Resource Simulation: CPU usage via psutil, energy (( E_c )), and latency (( L_{\text{local}} )) computed per client.
- Accuracy Evaluation: Cloud tests the global model on a 100-image MNIST test set.
- Reward Calculation: Balances accuracy gain against energy and latency costs.
## Limitations
- Single Round: Executes one training round per run (no multi-round loop yet).
- Fixed Participation: All clients participate every round (no dropout or selection).
- Simple Model: Small CNN (1 conv layer, 4 filters) limits accuracy potential.
- Static Resources: Fixed bandwidth (0.25 per client, 0.5 per edge) and CPU limits.
