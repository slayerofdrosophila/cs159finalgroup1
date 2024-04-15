## Pruning Filters for Efficient ConvNets

This code serves as a starting point for experimenting with filter pruning. The code is based off of the results of "Pruning Filters for Efficient ConvNets" by Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf
(2017). The code is designed to be modular and easy to extend for different architectures and datasets.

### Getting Started

This script requires the following python libraries:
* `argparse`
* `torchvision`

### Running the Script

The script can be run from the command line or the code can be tested in the provided Jupyter Notebook.

#### Command Line

1. Clone this repository or download the script.
2. Open your terminal and navigate to the directory containing the script.
3. Run the script with desired arguments. For example, to train a VGG16 model on CIFAR10 dataset for 100 epochs:

```bash
python main.py --action train --dataset CIFAR10 --model VGG --epochs 100
```

**Arguments:**

* `--action`: Required argument specifying the action to perform. Valid options are `train`, `prune`, and `test`.
* `--dataset`: Optional argument specifying the dataset to use. Defaults to CIFAR10. Other options include all datasets available in `torchvision.datasets`.
* `--model`: Optional argument specifying the model architecture. Defaults to VGG16. Other options are ResNet50.
* `--pretrained`: Optional argument (boolean) indicating whether to use pre-trained weights from PyTorch. Defaults to True.
* `--load-path`: Optional argument specifying the path to a file containing weights to load.
* `--save-path`: Optional argument specifying the path to save the trained model weights. Defaults to "model.pth".
* `--resume-epoch`: Optional argument specifying the epoch to resume training from. Defaults to -1 (start from scratch).

* Training arguments:
    * `--epochs`: Number of epochs to train/retrain the model. Defaults to 160.
    * `--batch-size`: Batch size for training. Defaults to 128.
    * `--lr`: Learning rate. Defaults to 0.001.
    * `--lr-decay`: Number of epochs after which to half the learning rate. Defaults to 10.
    * `--image-crop`: Size for random image cropping during training. Defaults to 32.
    * `--random-hflip`: Boolean argument indicating whether to randomly flip images horizontally during training. Defaults to True.

* Pruning arguments:
    * `--prune-retrain`: Boolean argument indicating whether to retrain the model after pruning. Defaults to False.
    * `--alternative-criteria`: Boolean argument (currently not functional) for an alternative pruning criteria. Defaults to False.
    * `--independent-prune-flag`: Boolean argument indicating whether to prune multiple layers independently. Defaults to False.
    * `--prune-layers`: Comma-separated list of layer indices for pruning.
    * `--prune-channels`: Comma-separated list specifying the number of channels to prune from each layer.
    * `--prune-stages` (ResNet only): Comma-separated list of "stage" indices for pruning in ResNet.

#### Jupyter Notebook

If you are using a Jupyter Notebook, you can set the arguments as environment variables before running the script. 

1. Set the `JUPYTER` environment variable to `"True"`.
2. Set the `JUPYTER_ARGS` environment variable to a space-separated list of arguments you want to pass to the script.
3. In your notebook, import the script and run it. The script will automatically parse the arguments from the environment variables.

### Example

This example trains a VGG16 model on the CIFAR10 dataset for 50 epochs:

```bash
python main.py --action train --dataset CIFAR10 --model VGG --epochs 50 
```

This example loads a pre-trained VGG-16 model, prunes filters, and retrains:

```bash
python main.py --action prune --load-path /pretrained_models/VGG16-CIFAR10-93.7%.pth --prune-layers conv1 conv2 conv8 conv9 conv10 conv11 conv12 conv13 --prune-channels 32 32 256 384 384 384 384 384 --lr 0.0001 --epochs 20
```

Then, to test a pruned model:
```bash
python main.py --action test --load-path model.pth --prune-layers conv1 conv2 conv8 conv9 conv10 conv11 conv12 conv13 --prune-channels 32 32 256 384 384 384 384 384
```


### Further Exploration

* Experiment with different pruning strategies (e.g., independent vs. global pruning).
* Try different pruning criteria (see the alternative pruning strategy in prune.py).
* Explore the impact of filter pruning on different network architectures and datasets.