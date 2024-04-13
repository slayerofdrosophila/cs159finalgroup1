import torch

from utils import args
from utils.model import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prune_network():
    model = get_model().to(device)
    model = prune_step(model, args.prune_layers, args.prune_channels, args.independent_prune_flag)

    return model


def prune_step(model, prune_layers, prune_channels, independent_prune_flag):
    model = model.cpu()

    count = 0  # count for indexing 'prune_channels'
    conv_count = 1  # conv count for 'indexing_prune_layers'
    dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    residue = None  # residue is need to prune by 'independent strategy'
    for i in range(len(model.features)):
        if isinstance(model.features[i], torch.nn.Conv2d):
            if dim == 1:
                new_, residue = get_new_conv(model.features[i], dim, channel_index, independent_prune_flag)
                model.features[i] = new_
                dim ^= 1

            if 'conv%d' % conv_count in prune_layers:
                channel_index = get_channel_index(model.features[i].weight.data, prune_channels[count], residue)
                new_ = get_new_conv(model.features[i], dim, channel_index, independent_prune_flag)
                model.features[i] = new_
                dim ^= 1
                count += 1
            else:
                residue = None
            conv_count += 1

        elif dim == 1 and isinstance(model.features[i], torch.nn.BatchNorm2d):
            new_ = get_new_norm(model.features[i], channel_index)
            model.features[i] = new_

    # update to check last conv layer pruned
    if 'conv13' in prune_layers:
        model.classifier[0] = get_new_linear(model.classifier[0], channel_index)

    return model


def get_channel_index(kernel, num_elimination, residue=None):
    # get candidate channel index for pruning
    # 'residue' is needed for pruning by 'independent strategy'

    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)

    vals, args = torch.sort(sum_of_kernel)

    return args[:num_elimination].tolist()


def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))

    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor


def get_new_conv(conv, dim, channel_index, independent_prune_flag=False):
    if dim == 0:
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

        new_conv.weight.data = index_remove(conv.weight.data, dim, channel_index)
        new_conv.bias.data = index_remove(conv.bias.data, dim, channel_index)

        return new_conv

    elif dim == 1:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - len(channel_index)),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

        new_weight = index_remove(conv.weight.data, dim, channel_index, independent_prune_flag)
        residue = None
        if independent_prune_flag:
            new_weight, residue = new_weight
        new_conv.weight.data = new_weight
        new_conv.bias.data = conv.bias.data

        return new_conv, residue


def get_new_norm(norm, channel_index):
    new_norm = torch.nn.BatchNorm2d(num_features=int(norm.num_features - len(channel_index)),
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)

    new_norm.weight.data = index_remove(norm.weight.data, 0, channel_index)
    new_norm.bias.data = index_remove(norm.bias.data, 0, channel_index)

    if norm.track_running_stats:
        new_norm.running_mean.data = index_remove(norm.running_mean.data, 0, channel_index)
        new_norm.running_var.data = index_remove(norm.running_var.data, 0, channel_index)

    return new_norm


def get_new_linear(linear, channel_index):
    new_linear = torch.nn.Linear(in_features=int(linear.in_features - len(channel_index)),
                                 out_features=linear.out_features,
                                 bias=linear.bias is not None)
    new_linear.weight.data = index_remove(linear.weight.data, 1, channel_index)
    new_linear.bias.data = linear.bias.data

    return new_linear
