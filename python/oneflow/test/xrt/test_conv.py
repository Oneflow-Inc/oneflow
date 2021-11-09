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

from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


def get_graph(device, weight):
    flow_device = flow.device(device)
    conv = flow.nn.Conv2d(1, 1, (1, 1), bias=False).to(flow_device)
    conv.weight = flow.nn.Parameter(flow.Tensor(weight))
    conv.to(flow_device)

    class Conv2DGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.conv = conv

        def build(self, x):
            return self.conv(x)

    conv2d_g = Conv2DGraph()
    return conv2d_g


class TestXrtConv2D(flow.unittest.TestCase):
    def test_xrt_conv2d(test_case):
        x = np.ones([1, 1, 1, 1], dtype=np.float32)
        weight = np.random.random([1, 1, 1, 1]).astype(np.float32)
        x_cpu = flow.tensor(x, dtype=flow.float32, device=flow.device("cpu"))
        x_cuda = flow.tensor(x, dtype=flow.float32, device=flow.device("cuda"))

        conv2d_g = get_graph("cpu", weight)
        out = conv2d_g(x_cpu)

        conv2d_g_openvino = get_graph("cpu", weight)
        conv2d_g_openvino.config.set_xrt_use_openvino(True)
        out_openvino = conv2d_g_openvino(x_cpu)
        test_case.assertTrue(
            np.allclose(out.numpy(), out_openvino.numpy(), rtol=1e-3, atol=1e-4)
        )

        conv2d_g_tensorrt = get_graph("cuda", weight)
        conv2d_g_tensorrt.config.set_xrt_use_tensorrt(True)
        out_tensorrt = conv2d_g_tensorrt(x_cuda)
        test_case.assertTrue(
            np.allclose(out.numpy(), out_tensorrt.numpy(), rtol=1e-3, atol=1e-4)
        )

        conv2d_g_xla = get_graph("cuda", weight)
        conv2d_g_xla.config.set_xrt_use_xla_jit(True)
        out_xla = conv2d_g_xla(x_cuda)
        test_case.assertTrue(
            np.allclose(out.numpy(), out_xla.numpy(), rtol=1e-3, atol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()
