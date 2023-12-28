import torch


def do_paper_initialization(squeezenet) -> None:
    """
    Initialize weights as in authors Caffe implementation.
    Authors implementation: https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.0/train_val.prototxt
    Caffe initializers: https://jihongju.github.io/2017/05/10/caffe-filler/#xavier-filler
    PyTorch initializers: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_

    Convolutions in fire module were initialized using `xavier` filler.
    Biases were initialized as zeroes.
    Final convolution was initialized using N(0.0, 0.01).
    """
    for module in squeezenet.features.modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        n = module.in_channels + module.out_channels
        torch.nn.init.uniform_(module.weight, -((3.0 / n) ** 0.5), (3.0 / n) ** 0.5)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)

    # Initialize final convolution with N(0.0, 0.01).
    torch.nn.init.normal_(squeezenet.final_conv[0].weight, 0.0, 0.01)
    torch.nn.init.constant_(squeezenet.final_conv[0].bias, 0.0)
