from torch.nn import Module


def polyak_average(model1: Module, model2: Module, ratio: float):
    """
    Interpolates the weights of model1 and model2 via polyak averaging. Modifies model1 inplace.
    """
    params1 = model1.named_parameters()
    params2 = dict(model2.named_parameters())

    for name, param in params1:
        assert name in params2
        param.data.copy_(ratio * param.data + (1 - ratio) * params2[name].data)
