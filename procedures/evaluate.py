import time

import torch

from utils.data import get_dataloader
from utils.loss import compute_accuracy
from utils.model import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_network(model=None):
    """
    Load the model and evaluate its performance on the test dataset.
    
    Args:
        model (torch.nn.Module, optional): The model to test. If not provided, 
            the model is loaded according to program arguments.

    Returns:
        float: The accuracy of the model on the test dataset.
    """

    if model is None:
        model = get_model()
    model = model.to(device)
    dataloader = get_dataloader(training=False)

    time_start = time.time()
    accuracy = test_step(model, dataloader)
    time_end = time.time()

    print(f"Testing network ({time_end - time_start:.2f}), \tAccuracy: {accuracy:.2f}")


def test_step(model, dataloader):
    """
    Tests the model on the dataloader and returns the accuracy.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.

    Returns:
        float: The accuracy of the model over each batch
    """

    model.eval()

    total_accuracy = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_accuracy += compute_accuracy(outputs, targets)

    return total_accuracy / len(dataloader)
