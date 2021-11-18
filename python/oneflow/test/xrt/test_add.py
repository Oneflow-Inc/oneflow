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
import oneflow.unittest


def get_graph():
    class AddGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, x, y):
            return flow.add(x, y)

    add_g = AddGraph()
    return add_g


class TestXrtAdd(flow.unittest.TestCase):
    def test_xrt_add(test_case):
        x = np.random.random((1, 10, 2)).astype(np.float32)
        y = np.random.random((1, 10, 2)).astype(np.float32)
        x_cpu = flow.tensor(x, dtype=flow.float32, device=flow.device("cpu"))
        x_cuda = flow.tensor(x, dtype=flow.float32, device=flow.device("cuda"))
        y_cpu = flow.tensor(y, dtype=flow.float32, device=flow.device("cpu"))
        y_cuda = flow.tensor(y, dtype=flow.float32, device=flow.device("cuda"))

        add_g = get_graph()
        out = add_g(x_cpu, y_cpu)

        add_g_openvino = get_graph()
        add_g_openvino.config.enable_xrt_use_openvino(True)
        out_openvino = add_g_openvino(x_cpu, y_cpu)
        test_case.assertTrue(
            np.allclose(out.numpy(), out_openvino.numpy(), rtol=1e-3, atol=1e-4)
        )

        add_g_tensorrt = get_graph()
        add_g_tensorrt.config.enable_xrt_use_tensorrt(True)
        out_tensorrt = add_g_tensorrt(x_cuda, y_cuda)
        test_case.assertTrue(
            np.allclose(out.numpy(), out_tensorrt.numpy(), rtol=1e-3, atol=1e-4)
        )

        add_g_xla = get_graph()
        add_g_xla.config.enable_xrt_use_xla_jit(True)
        out_xla = add_g_xla(x_cuda, y_cuda)
        test_case.assertTrue(
            np.allclose(out.numpy(), out_xla.numpy(), rtol=1e-3, atol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()
