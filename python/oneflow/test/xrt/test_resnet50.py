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

import unittest
from random import randint
from random import choice

import numpy as np

from oneflow.test.modules.resnet50_model import resnet50
import oneflow as flow
from test_xrt import *


res50_module = resnet50(
    replace_stride_with_dilation=[False, False, False], norm_layer=flow.nn.BatchNorm2d,
).eval()


def get_cpu_graph():
    class ResNet50Graph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.res50_module = res50_module

        def build(self, x):
            return self.res50_module(x)

    restnet50_g = ResNet50Graph()
    return restnet50_g


def get_cuda_graph():
    class ResNet50Graph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.res50_module = res50_module.to(device="cuda")

        def build(self, x):
            return self.res50_module(x)

    restnet50_g = ResNet50Graph()
    return restnet50_g


class TestXrtResNet50(flow.unittest.TestCase):
    def test_xrt_restnet50(test_case):
        x = np.random.random((1, 3, 224, 224)).astype(np.float32)
        x_cpu = flow.tensor(x, dtype=flow.float32, device=flow.device("cpu"))
        x_cuda = flow.tensor(x, dtype=flow.float32, device=flow.device("cuda"))

        resnet50_g = get_cpu_graph()
        out = resnet50_g(x_cpu)

        test_xrt_openvino(test_case, get_cpu_graph(), x_cpu, out)
        test_xrt_tensorrt(test_case, get_cuda_graph(), x_cuda, out)
        test_xrt_xla(test_case, get_cuda_graph(), x_cuda, out)


if __name__ == "__main__":
    unittest.main()
