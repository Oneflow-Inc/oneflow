import torch
from torch import nn

from oneflow.python.test.onnx.load.util import load_pytorch_module_and_check


def test_flatten(test_case):
    class Net(nn.Module):
        def forward(self, x):
            x = torch.flatten(x, 1)
            return x
    load_pytorch_module_and_check(test_case, Net)
