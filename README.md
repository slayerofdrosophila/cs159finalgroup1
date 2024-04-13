# Pruning Filters For Efficient ConvNets

Train the model
```python main.py --action train```

Prune the model and retrain it
```python main.py --action prune```

Evaluate the model
```python main.py --action test```

#### Absolute sum of filter weights for each layer of VGG-16 trained on CIFARA-10
* This graph was created in [jupyter notebook](https://github.com/tyui592/notepad/blob/master/pruning_filters_for_efficient_convets/prune_filter_for_efficient_convnets.ipynb). You can make the graph yourself.

![figure1](./imgs/figure1.png)

#### Pruning filters with the lowest absolute weights sum and their corresponding test accuracies on CIFAR-10
* This graph was created in [jupyter notebook](https://github.com/tyui592/notepad/blob/master/pruning_filters_for_efficient_convets/prune_filter_for_efficient_convnets.ipynb). You can make the graph yourself.

![figure2](./imgs/figure2.png)

#### Prune and retrain for each single layer of VGG-16 on CIFAR-10
* This graph was created in [jupyter notebook](https://github.com/tyui592/notepad/blob/master/pruning_filters_for_efficient_convets/prune_filter_for_efficient_convnets.ipynb). You can make the graph yourself.

![figure3](./imgs/figure3.png)
