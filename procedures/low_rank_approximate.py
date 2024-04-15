import torch
import torch.nn as nn

from utils.model import get_model
from utils import args

import torch
import torch.nn as nn

from utils.model import get_model
from utils import args
from procedures.evaluate import test_network


def apply_low_rank_approximation():
    def decompose_linear(layer, compression_ratio):
        original_rank = min(layer.weight.data.size())
        target_rank = int(original_rank * compression_ratio)  # Calculate the new rank
        U, S, V = torch.svd_lowrank(layer.weight.data, q=target_rank)
        linear1 = nn.Linear(layer.in_features, target_rank, bias=False)
        linear1.weight.data = torch.mm(torch.diag(S), V.t())
        linear2 = nn.Linear(
            target_rank,
            layer.out_features,
            bias=True if layer.bias is not None else False,
        )
        linear2.weight.data = U
        if layer.bias is not None:
            linear2.bias.data = layer.bias.data
        return nn.Sequential(linear1, linear2)

    def decompose_conv2d(layer, compression_ratio):
        # Placeholder: Adapt this based on the specifics of Conv2d weight decomposition
        return layer

    def _apply_low_rank_approximation(model, compression_ratio):
        for name, child in list(model.named_children()):
            if isinstance(child, nn.Linear):
                setattr(model, name, decompose_linear(child, compression_ratio))
            elif isinstance(child, nn.Conv2d):
                setattr(model, name, decompose_conv2d(child, compression_ratio))

            _apply_low_rank_approximation(child, compression_ratio)

    model = get_model().to("cpu")
    compression_ratio = 0.5  # Reduce to 50% of the original rank
    print(f"Applying low-rank approximation with compression ratio of {compression_ratio}...")
    _apply_low_rank_approximation(model, compression_ratio)
    print("Low-rank approximation applied. Testing the model...")
    test_network(model)
    torch.save({"state_dict": model.state_dict()}, args.save_path)
    print(f"Low-rank approximated model saved to {args.save_path}")
