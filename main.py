from parameter import get_parameter
from train import train_network
from evaluate import test_network
from prune import prune_network

import time

if __name__ == '__main__':
    args = get_parameter()

    start_time = time.time()

    network = None
    if args.train_flag:
        network = train_network(args)

    end_time = time.time()

    print("TIME FOR TRAIN/FINETUNE:", end_time - start_time)

    if args.prune_flag:
        network = prune_network(args, network=network)

    test_network(args, network=network)
