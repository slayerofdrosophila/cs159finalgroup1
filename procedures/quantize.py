"""Code for quantizing the weights of a PyTorch model."""

import torch
import torch.quantization
import torch.nn as nn

from utils import args
from procedures.evaluate import test_network


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.quantized.engine = "qnnpack"


def quantize_network():
    # Avoids a circular import with get_model() using the pruning routine
    from utils.model import get_model

    model = get_model().to(device)

    print("Quantizing the model to int8...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # the original model
        {nn.Linear, nn.Conv2d, nn.BatchNorm2d},  # a set of layers to dynamically quantize
        dtype=torch.qint8,  # the data type to quantize to
    )
    print("Successfully quantized the model to int8. Testing the model...")
    test_network(quantized_model)
    torch.save({"state_dict": model.state_dict()}, args.save_path)
    print(f"Quantized model saved to {args.save_path}")
