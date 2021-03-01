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

# from absl import app
# from absl.testing import absltest


def test_pad(test_case):
    class Net(nn.Module):
        def forward(self, x):
            x = F.pad(x, (2, 3))
            return x

    load_pytorch_module_and_check(test_case, Net)


def test_pad_with_value(test_case):
    class Net(nn.Module):
        def forward(self, x):
            x = F.pad(x, (2, 3), value=3.5)
            return x

    load_pytorch_module_and_check(test_case, Net)


# test_case = absltest.TestCase
# test_pad(test_case)
