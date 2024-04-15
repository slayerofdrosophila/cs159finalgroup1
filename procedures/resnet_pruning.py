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

    model = exec_pruning(model, args.prune_stages, args.prune_layers, args.prune_channels)

    return model



def exec_pruning(model, stages_to_prune=[1,2,3,4], layers_to_prune=[1], num_channels_to_prune=[10]):
    model = model.cpu()

    print("Mapping model")
    mapping = create_model_map(model)

    print("Preparing indices")
    layer_indices = get_layer_indices(stages_to_prune, ["*"], layers_to_prune, mapping)

    print(layer_indices)

    print("On to pruning!")
    for i in layer_indices:
        print(i)
        for j in i:
            print(j)
            for k in j:
                print(k)
                num_channels_to_prune = 5
                model = prune_this_layer(model, k, num_channels_to_prune)

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
                    if layer_in_block_name in ("conv1", "conv2", "conv3", "bn1", "bn2", "bn3"):
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


def prune_this_layer(model, layer_idx, channels):

    new_channels = 0
    for idx, tuple in enumerate(model.named_modules()):
        name, module = tuple
        
        if idx == layer_idx:

            # Find how many out channels there are, replace it, etc.
            in_channels = module.in_channels
            out_channels = module.out_channels - channels  # Reducing the number of filters by half
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            bias = module.bias is not None

            # Remove half the old filters... No cireteria rn
            orig_filters = module.weight.clone()
            new_filters = orig_filters[:out_channels, :, :, :]

            # Creating a new layer with modified parameters
            new_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            new_layer.weight.data = new_filters

            # Replace the original layer with the new one
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_layer)


        # Replace bnorm layer below it with new dimensions
        if idx == layer_idx + 1:

            new_bn_layer = torch.nn.BatchNorm2d(out_channels)

            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_bn_layer)


        # Must unfortunately replace the layer underneath THAT with new conv layer with correct amount of input channels
        if idx == layer_idx + 2:

            in_channels = module.in_channels - channels 
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

            return model


