import torch

import time

from procedures.evaluate import test_network
from procedures.prune import prune_network
from procedures.train import train_network
from utils import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    print(f"Device: {device}")

    start_time = time.time()
    print(f"{'-*-' * 10}\n\tRunning: {args.action}\n{'-*-' * 10}")

    if args.action == 'train':
        train_network()
    elif args.action == 'prune':
        model = prune_network()
        train_network(model)
    elif args.action == 'test':
        test_network()

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
