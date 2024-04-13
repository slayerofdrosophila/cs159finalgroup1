import time

import torch

from utils.data import get_dataloader
from utils.loss import compute_accuracy
from utils.model import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_network(model=get_model()):
    model = model.to(device)
    dataloader = get_dataloader()

    time_start = time.time()
    accuracy = test_step(model, dataloader)
    time_end = time.time()

    print(f"Testing network ({time_end - time_start:.2f}), \tAccuracy: {accuracy:.2f}")


def test_step(model, dataloader):
    model.eval()

    total_accuracy = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_accuracy += compute_accuracy(outputs, targets)

    return total_accuracy / len(dataloader)
