import torch

from utils import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prune_network():
    """
    Prunes the specified layers and channels from the model.
    
    Returns:
        torch.nn.Module: The pruned model.
    """
    from utils.model import get_model  # Avoids a circular import with get_model() using the pruning routine

    model = get_model().to(device)

    print(model)
    model = prune_step(model, args.prune_layers, args.prune_channels, args.independent_prune_flag)
    print(model)

    return model


def prune_step(model, prune_layers, prune_channels, independent_prune_flag):
    """
    Prunes the specified layers and channels from the model.

    Args:
        model (torch.nn.Module): The model to be pruned.
        prune_layers (list): List of layer names to be pruned.
        prune_channels (list): Number of channels to prune in each layer.
        independent_prune_flag (bool): Flag indicating whether to apply independent pruning strategy.

    Returns:
        torch.nn.Module: The pruned model.
    """
    model = model.cpu()

    count = 0  # count for indexing 'prune_channels'
    conv_count = 1  # conv count for 'indexing_prune_layers'
    dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    residue = None  # residue is need to prune by 'independent strategy'
    for i in range(len(model.features)):
        # Check if current layer is convolutional
        if isinstance(model.features[i], torch.nn.Conv2d):
            if dim == 1:
                new_, residue = get_new_conv(model.features[i], dim, channel_index, independent_prune_flag)
                model.features[i] = new_
                # Toggle dim to switch to pruning along input channels in the next layer
                dim ^= 1
            # Check if the current convolutional layer is marked for pruning
            if 'conv%d' % conv_count in prune_layers:
                # Calculate indices of channels to prune based on kernel weights
                channel_index = get_channel_index(model.features[i].weight.data, prune_channels[count], residue)
                new_ = get_new_conv(model.features[i], dim, channel_index, independent_prune_flag)
                model.features[i] = new_
                # Toggle dim to switch to pruning along input channels in the next layer
                dim ^= 1
                count += 1
            else:
                residue = None
            conv_count += 1
        # Check if current layer is batch normalization and dim indicates pruning along output channels
        elif dim == 1 and isinstance(model.features[i], torch.nn.BatchNorm2d):
            new_ = get_new_norm(model.features[i], channel_index)
            model.features[i] = new_

    # update to check last conv layer pruned
    if 'conv13' in prune_layers:
        model.classifier[0] = get_new_linear(model.classifier[0], channel_index)

    return model


def get_channel_index(kernel, num_elimination, residue=None):
    """
    Get the indices of channels to be pruned based on the sum of absolute kernel weights.

    Args:
        kernel (torch.Tensor): The kernel weights tensor.
        num_elimination (int): The number of channels to prune.
        residue (torch.Tensor): The residue tensor used for independent pruning strategy.

    Returns:
        list: The indices of channels to be pruned.
    """
    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)

    vals, args = torch.sort(sum_of_kernel)

    return args[:num_elimination].tolist()


def index_remove(tensor, dim, index, removed=False):
    """
    Remove specified indices along a given dimension of a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.
        dim (int): Dimension along which to remove indices.
        index (list): Indices to be removed.
        removed (bool, optional): Flag to return removed elements.

    Returns:
        torch.Tensor: Tensor with specified indices removed.
        torch.Tensor: Removed elements (if `removed=True`).
    """
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
    """
    Create a new convolutional layer with pruned channels.

    Args:
        conv (torch.nn.Conv2d): Original convolutional layer.
        dim (int): Dimension along which to prune (0: output channels, 1: input channels).
        channel_index (list): Indices of channels to be pruned.
        independent_prune_flag (bool, optional): Flag for independent pruning strategy.

    Returns:
        torch.nn.Conv2d: New convolutional layer.
        torch.Tensor: Residual weights for independent pruning strategy (if applicable).
    """
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
    """
    Create a new batch normalization layer with pruned channels.

    Args:
        norm (torch.nn.BatchNorm2d): Original batch normalization layer.
        channel_index (list): Indices of channels to be pruned.

    Returns:
        torch.nn.BatchNorm2d: New batch normalization layer.
    """
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
    """
    Create a new fully connected layer with pruned input features.

    Args:
        linear (torch.nn.Linear): Original fully connected layer.
        channel_index (list): Indices of input features to be pruned.

    Returns:
        torch.nn.Linear: New fully connected layer.
    """
    new_linear = torch.nn.Linear(in_features=int(linear.in_features - len(channel_index)),
                                 out_features=linear.out_features,
                                 bias=linear.bias is not None)
    new_linear.weight.data = index_remove(linear.weight.data, 1, channel_index)
    new_linear.bias.data = linear.bias.data

    return new_linear

def prune_resnet(model):
    """
    Prune a ResNet model by reducing the number of filters in convolutional layers with least L1-norm.

    Args:
        model (torch.nn.Module): ResNet model to be pruned.

    Returns:
        torch.nn.Module: Pruned ResNet model.
    """
    model = model.cpu()

    notatthebeginning = False
    for name, module in model.named_modules():

        # Check if the module is a Conv2d layer within a Bottleneck block's 'conv1'
        if isinstance(module, torch.nn.Conv2d) and 'conv1' in name and module.kernel_size == (1,1):

            notatthebeginning = True

            print("Original:", module)

            # Cut half the channels
            in_channels = module.in_channels
            out_channels = module.out_channels // 2  # Reducing the number of filters by half
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            bias = module.bias is not None

            print("Out channels", module.out_channels)

            # Remove half the old filters... No cireteria rn
            orig_filters = module.weight.clone()
            new_filters = orig_filters[:out_channels, :, :, :]

            # Creating a new layer with modified parameters
            new_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            new_layer.weight.data = new_filters

            # Replace the original layer with the new one
            print(name)
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_layer)
            print(f"Modified: {new_layer}")
            print("New out channels", out_channels)

        # Remake layer below
        if isinstance(module, torch.nn.Conv2d) and 'conv2' in name:
            notatthebeginning = True

            # Remake layer below in order to fit
            print(f"Original: {module}")

            in_channels = module.in_channels // 2 
            out_channels = module.out_channels 
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            bias = module.bias is not None

            # Creating a new layer with modified parameters
            new_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

            # Sadly, layer must be clean slate... i thnk
            # Initialize the new layer (example with Xavier uniform)
            torch.nn.init.xavier_uniform_(new_layer.weight)

            # Replace the original layer with the new one
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_layer)
            print(f"Modified: {new_layer}")

        # Remake batchnorm layer between the two
        if isinstance(module, torch.nn.BatchNorm2d) and 'bn1' in name and notatthebeginning:
            # Assuming the convolutional layer's output channels were modified to `new_out_channels`
            # Get the number of features from the conv layer that feeds into this bn layer
            conv_name = name.replace('bn1', 'conv1')
            conv_layer = dict(model.named_modules())[conv_name]
            new_out_channels = conv_layer.out_channels  # Adjusted in previous steps


            # Create a new BatchNorm2d layer
            new_bn_layer = torch.nn.BatchNorm2d(new_out_channels)

            # Replace the original bn layer with the new one
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_bn_layer)
            print(f"Updated BatchNorm layer {name} to new out_channels: {new_out_channels}")