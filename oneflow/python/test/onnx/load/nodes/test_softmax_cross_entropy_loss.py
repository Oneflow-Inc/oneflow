"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
from torch import nn
import torch.nn.functional as F

from oneflow.python.test.onnx.load.util import load_pytorch_module_and_check


def test_softmax_cross_entropy_loss_default(test_case):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.loss = nn.CrossEntropyLoss()
            self.register_buffer("label", torch.tensor([0, 1], dtype=torch.int64))

        def forward(self, x):
            x = self.loss(x, self.label)
            return x

    load_pytorch_module_and_check(test_case, Net, input_size=(2, 4))


def test_softmax_cross_entropy_loss_sum(test_case):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.loss = nn.CrossEntropyLoss(reduction="sum")
            self.register_buffer("label", torch.tensor([0, 1], dtype=torch.int64))

        def forward(self, x):
            x = self.loss(x, self.label)
            return x

    load_pytorch_module_and_check(test_case, Net, input_size=(2, 4))


def test_softmax_cross_entropy_loss_none(test_case):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.loss = nn.CrossEntropyLoss(reduction="none")
            self.register_buffer("label", torch.tensor([0, 1], dtype=torch.int64))

        def forward(self, x):
            x = self.loss(x, self.label)
            return x

    load_pytorch_module_and_check(test_case, Net, input_size=(2, 4))
