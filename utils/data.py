import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose, ToImage, ToDtype

from utils import args


class NoneTransform(object):
    """Dummy transform that does nothing"""

    def __call__(self, image):
        return image


def get_normalizer(dataset):
    """
    Returns a normalizer according to the precalculated statistics of the dataset.
    Only CIFAR10, CIFAR100, and ImageNet are currently included here.
    """

    if dataset == 'CIFAR10':
        return Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dataset == 'CIFAR100':
        return Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    elif dataset == 'ImageNet':
        return Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        print("No normalizer defined for this data set")
        return NoneTransform()


def get_transformer(training=True):
    """Returns a transformer for the dataset, with augmentations if training=True."""

    transformers = []

    if training:
        if args.image_crop:
            transformers.append(RandomCrop(args.image_crop, padding=4))
        if args.random_hflip:
            transformers.append(RandomHorizontalFlip())

    transformers.append(Compose([ToImage(), ToDtype(torch.float32, scale=True)]))
    transformers.append(get_normalizer(args.dataset))

    return Compose(transformers)


def get_dataloader(training=True):
    """Returns a DataLoader for the dataset, according to program arguments"""

    dataset = (torchvision.datasets.__dict__[args.dataset]
               (root="/data", train=training, transform=get_transformer(training), download=True))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    return dataloader
