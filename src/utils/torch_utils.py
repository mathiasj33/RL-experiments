import torch
from torch import nn
from torch.nn import Module

printed_device = False

def get_device():
    global printed_device

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'
    if not printed_device:
        print(f'Using device: {device_name}')
        printed_device = True
    return device


def make_mlp_layers(layer_sizes: list[int], activation: any) -> list[nn.Module]:
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i != len(layer_sizes) - 2:
            layers.append(activation())
    return layers


def polyak_average(model1: Module, model2: Module, ratio: float):
    """
    Interpolates the weights of model1 and model2 via polyak averaging. Modifies model1 inplace.
    """
    params1 = model1.named_parameters()
    params2 = dict(model2.named_parameters())

    for name, param in params1:
        assert name in params2
        param.data.copy_(ratio * param.data + (1 - ratio) * params2[name].data)
