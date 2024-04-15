import argparse
import os

import torchvision

parser = argparse.ArgumentParser(prog="Pruning Filters for Efficient ConvNets",
                                 description="This program helps demonstrate filter pruning.")
parser.add_argument('--action', choices=['train', 'prune', 'test'], required=True, help="Action to perform")
parser.add_argument('--dataset', choices=torchvision.datasets.__all__, default='CIFAR10', help="Dataset to use")
parser.add_argument('--model', choices=['VGG', 'ResNet'], default='VGG', help="Model to use, either VGG16 or ResNet50")

parser.add_argument('--pretrained', type=bool, default=True, help="Use PyTorch's pretrained weights")
parser.add_argument('--load-path', type=str, help="Load weights from an existing file")
parser.add_argument('--save-path', type=str, default="model.pth", help="Save weights to a file")
parser.add_argument('--resume-epoch', type=int, default=-1, help="Epoch to resume training")

parser.add_argument('--epochs', type=int, default=160, help="Number of epochs to train/retrain the model")
parser.add_argument('--batch-size', type=int, default=128, help="Batch size for training")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--lr-decay', type=int, default=10, help="Half the learning rate every n epochs")
parser.add_argument('--image-crop', type=int, default=32, help="Size for random image cropping in training")
parser.add_argument('--random-hflip', action='store_true', default=True,
                    help="Randomly flip images horizontally in training")

parser.add_argument('--prune-retrain', action='store_true', default=False)
parser.add_argument('--independent-prune-flag', action='store_true', default=False,
                    help='prune multiple layers by "independent strategy"')
parser.add_argument('--prune-layers', nargs='+', help='layer index for pruning', default=None)
parser.add_argument('--prune-channels', nargs='+', type=int, help='number of channel to prune layers', default=None)

# If using Jupyter Notebook, provide the arguments as environment variables
if os.environ.get('JUPYTER') == 'True':
    args = parser.parse_args(os.environ.get('JUPYTER_ARGS').split())
else:
    args = parser.parse_args()

action: str = args.action
dataset: str = args.dataset
model: str = args.model

pretrained: bool = args.pretrained
load_path: str = args.load_path
save_path: str = args.save_path
resume_epoch: int = args.resume_epoch

epochs: int = args.epochs
batch_size: int = args.batch_size
lr: float = args.lr
lr_decay: int = args.lr_decay
image_crop: int = args.image_crop
random_hflip: bool = args.random_hflip

prune_retrain: bool = args.prune_retrain
independent_prune_flag: bool = args.independent_prune_flag
prune_layers: list = args.prune_layers
prune_channels: list = args.prune_channels
