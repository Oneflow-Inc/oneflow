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

import oneflow as flow
from test_xrt import *


def get_graph(device):
    flow_device = flow.device(device)
    m = flow.nn.LeakyReLU(0.1)
    m.to(flow_device)

    class LeakyReLUGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = m

        def build(self, x):
            return m(x)

    leaky_relu_g = LeakyReLUGraph()
    return leaky_relu_g


class TestXrtLeakyReLU(flow.unittest.TestCase):
    def test_xrt_leaky_relu(test_case):
        x = np.random.random((1, 10, 2)).astype(np.float32)
        x_cpu = flow.tensor(x, dtype=flow.float32, device=flow.device("cpu"))
        x_cuda = flow.tensor(x, dtype=flow.float32, device=flow.device("cuda"))

        leaky_relu_g = get_graph("cpu")
        out = leaky_relu_g(x_cpu)

        test_xrt_openvino(test_case, get_graph("cpu"), x_cpu, out)
        test_xrt_tensorrt(test_case, get_graph("cuda"), x_cuda, out)
        test_xrt_xla(test_case, get_graph("cuda"), x_cuda, out)


if __name__ == "__main__":
    unittest.main()
