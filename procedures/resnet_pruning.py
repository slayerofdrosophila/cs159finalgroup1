import torch

from utils import args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



"""
Necessary args: prune layers, prune channels, prune stages
"""
def prune_resnet(model):

    args.prune_stages = list(map(int, args.prune_stages))
    args.prune_layers = list(map(int, args.prune_layers))
    args.prune_channels = list(map(int, args.prune_channels))

    if args.prune_blocks != None:
        args.prune_blocks = list(map(int, args.prune_blocks))

    model = exec_pruning(model, stages_to_prune=args.prune_stages, blocks_to_prune=args.prune_blocks, layers_to_prune=args.prune_layers, channels_to_prune=args.prune_channels)

    return model

"""
Can control:
Stages, layers
"""
def exec_pruning(model, stages_to_prune=[1,2,3,4], blocks_to_prune=["*"], layers_to_prune=[1], channels_to_prune=[10]):
    model = model.cpu()

    print("Mapping model")
    mapping = create_model_map(model)

    if blocks_to_prune == None:
        blocks_to_prune = ["*"]

    print("Preparing indices")
    print("stages to prune:", stages_to_prune)
    print("blocks to prune:", blocks_to_prune)
    print("layers to prune:", layers_to_prune)
    layer_indices = get_layer_indices(stages_to_prune, blocks_to_prune, layers_to_prune, mapping)

    print(layer_indices)

    print("On to pruning!")

    for list_idx, layer_num in enumerate(layer_indices):
        print("layer map",layer_num)
        for block_map in layer_num:
            print("block map", block_map)
            print("trimming this layer", block_map[list_idx], "num channels", channels_to_prune[list_idx])
            # block_map[list_idx] is the corresponding layer -- we will do just ONE layer per block, the one at this index
            model = prune_this_layer(model, block_map[list_idx], channels_to_prune[list_idx])

    return model



"""
Organization:
List for each stage
Goes like [[[conv1, bn1, conv2, bn2...], [block2 indices]], [[stage2 block1], [stage2 block2]], [stage3 stuff]]

Top level: 4 elements, 1 for each stage
Second level: N elements, 1 for each block
Third level: 6 elements, 1 for each layer of interest
"""
def create_model_map(model):



    model_mapping = []
    idx = 0

    for name, pytorch_module in model.named_modules():

        if len(name.split("layer")) > 1:
            # print("full module name", name)

            stage_number = int(name.split("layer")[1].split(".")[0])
            # print("stage", stage_number)

            # Add entry in list for this layer, if needed
            if len(model_mapping) < stage_number:
                model_mapping.append([])
            
            # If this is inside a block, not the high order Sequence item
            if len(name.split(".")) > 1:
                block_num = int(name.split(".")[1])
                # print("block", block_num)

                # Add list for this block, if needed
                if (len(model_mapping[stage_number-1])) < block_num + 1:
                    model_mapping[stage_number-1].append([])

                # If this is a layer, not a high order Bottleneck item
                if len(name.split(".")) > 2:
                    layer_in_block_name = name.split(".")[2]
                    # print("layer",layer_in_block_name)

                    # If this is a layer we wish to potentially modify
                    if layer_in_block_name in ("conv1", "conv2", "conv3"):
                        # print("Adding this layer name:", layer_in_block_name)
                        # print(model_mapping)
                        model_mapping[stage_number-1][block_num].append(idx)
        else:
            # print("Not inside a stage", name)
            pass
        idx += 1
        # print("=-==-=-=-=-=-")

    return model_mapping
        



"""
[[layeridx], [[layersidxs] for each block, for each stage]
"""
def get_layer_indices(stage_idxs, block_idxs, layer_idxs, D):

    print(block_idxs)

    stage_idxs = [a - 1 for a in stage_idxs]
    if block_idxs[0] != "*":
        block_idxs = [b - 1 for b in block_idxs]
    layer_idxs = [b - 1 for b in layer_idxs]

    result = []
    for a in stage_idxs:
        list_a = D[a]
        selected_items_a = []

        block_idxs = range(len(list_a)) if block_idxs[0] == "*" else block_idxs

        for b in block_idxs:
            sublist_b = list_a[b]
            selected_items_b = []
            for c in layer_idxs:
                selected_items_b.append(sublist_b[c])
            selected_items_a.append(selected_items_b)
        result.append(selected_items_a)
    return result


def prune_this_layer(model, layer_idx, channels_to_remove):

    new_channels = 0
    for idx, tuple in enumerate(model.named_modules()):
        name, module = tuple
        
        if idx == layer_idx:

            # Find how many out channels there are, replace it, etc.
            in_channels = module.in_channels
            out_channels = module.out_channels - channels_to_remove  # Reducing the number of filters by half
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            bias = module.bias is not None

            # Remove half the old filters... No cireteria rn
            orig_filters = module.weight.clone()
            # new_filters = orig_filters[:out_channels, :, :, :]
            prune_indices = get_channel_index(orig_filters, channels_to_remove)

            mask = torch.ones(orig_filters.size(0), dtype=torch.bool)
            mask[prune_indices] = False
            new_filters = orig_filters[mask]


            # Creating a new layer with modified parameters
            new_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            new_layer.weight.data = new_filters

            # Replace the original layer with the new one
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_layer)


        # Replace bnorm layer below it with new dimensions
        if idx == layer_idx + 1:

            mask[prune_indices] = False
            
            assert isinstance(module, torch.nn.BatchNorm2d), "bnorm must be an instance of torch.nn.BatchNorm2d"
            # Prune the batch normalization layer
            new_bn_layer = torch.nn.BatchNorm2d(num_features=out_channels)
            new_bn_layer.weight.data = module.weight.data[mask]
            new_bn_layer.bias.data = module.bias.data[mask]
            new_bn_layer.running_mean = module.running_mean[mask]
            new_bn_layer.running_var = module.running_var[mask]
    
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_bn_layer)


        # Must unfortunately replace the layer underneath THAT with new conv layer with correct amount of input channels
        if idx == layer_idx + 2:

            in_channels = module.in_channels - channels_to_remove 
            out_channels = module.out_channels 
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            bias = module.bias is not None

            # Creating a new layer with modified parameters
            new_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

            # Copying weights from the original layer -- But only some amount
            # new_weights = module.weight[:, 0:in_channels, :, :]
            new_weights = index_remove(module.weight, 1, prune_indices)
            

            # Setting the weights and bias of the new layer
            new_layer.weight = torch.nn.Parameter(new_weights)

            # Replace the original layer with the new one
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_layer)

            return model



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


def get_channel_index(kernel, num_elimination, residue=None):
    # get candidate channel index for pruning
    # 'residue' is needed for pruning by 'independent strategy'

    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)

    vals, args = torch.sort(sum_of_kernel)

    return args[:num_elimination].tolist()