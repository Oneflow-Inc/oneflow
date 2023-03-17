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
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_reshape_forward(test_case, shape, device, dtype):
    arr = np.random.randn(*shape)
    x1 = flow.tensor(arr, device=flow.device(device), dtype=dtype)
    x2 = flow.tensor(arr, device="cpu", dtype=dtype)
    mlu_out = flow.reshape(x1, shape=shape)
    cpu_out = flow.reshape(x2, shape=shape)
    test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out, 0.0001, 0.0001))

    class ReshapeGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, x):
            return flow.reshape(x, shape=shape)

    graph = ReshapeGraph()
    graph_out = graph(x1)
    test_case.assertTrue(np.allclose(graph_out.numpy(), cpu_out, 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestReshapeCambriconModule(flow.unittest.TestCase):
    def test_reshape(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_reshape_forward,
        ]
        arg_dict["shape"] = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 3, 4, 5, 6)]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
            flow.int8,
            flow.uint8,
            flow.int32,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_0_size_tensor_reshape(test_case):
        arr = np.random.randn(4, 2, 0, 3)
        x1 = flow.tensor(arr, device="mlu", dtype=flow.float32)
        x2 = flow.tensor(arr, device="cpu", dtype=flow.float32)
        mlu_out = flow.reshape(x1, shape=(3, 0, 3))
        cpu_out = flow.reshape(x2, shape=(3, 0, 3))
        test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out, 0.0001, 0.0001))

    def test_0_dim_tensor_reshape(test_case):
        x1 = flow.tensor(2.0, device="mlu", dtype=flow.float32)
        x2 = flow.tensor(2.0, device="cpu", dtype=flow.float32)
        mlu_out = flow.reshape(x1, shape=(1, 1, 1))
        cpu_out = flow.reshape(x2, shape=(1, 1, 1))
        test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out, 0.0001, 0.0001))


if __name__ == "__main__":
    unittest.main()
