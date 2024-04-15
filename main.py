"""
This program helps demonstrate filter pruning by providing a basic implementation of the paper
"Pruning Filters for Efficient ConvNets," ICLR 2017. Use the --action argument to either train,
prune, or test a model. See utils/args.py for the available arguments.
"""

import time

from procedures.evaluate import test_network
from procedures.prune import prune_network
from procedures.train import train_network
from procedures.quantize import quantize_network
from procedures.low_rank_approximate import apply_low_rank_approximation


from utils import args

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(f"Device: {device}")

    start_time = time.time()
    print(f"{'-*-' * 10}\n\tRunning: {args.action}\n{'-*-' * 10}")

    if args.action == "train":
        train_network()
    elif args.action == "prune":
        model = prune_network()
        if args.prune_retrain:
            train_network(model)
    elif args.action == "test":
        test_network()
    elif args.action == "quantize":
        quantize_network()
    elif args.action == "low-rank-approximate":
        apply_low_rank_approximation()

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
