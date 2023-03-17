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


def _cpu_transpose(x, perm):
    return flow.transpose(x.cpu(), perm)


def _test_eager_transpose(test_case, device, shape, perm, dtype):
    arr = np.random.randn(*shape)
    x = flow.tensor(arr, device=flow.device(device), dtype=dtype)
    of_out = flow.transpose(x, perm)
    cpu_out = _cpu_transpose(x, perm)
    test_case.assertTrue(np.allclose(of_out.numpy(), cpu_out, 0.0001, 0.0001))


def _test_graph_transpose(test_case, device, shape, perm, dtype):
    arr = np.random.randn(*shape)
    x = flow.tensor(arr, device=flow.device(device), dtype=dtype)

    class TransposeGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, x):
            return flow.transpose(x, perm=perm)

    graph = TransposeGraph()
    graph_out = graph(x)
    cpu_out = _cpu_transpose(x, perm)
    test_case.assertTrue(np.allclose(graph_out.numpy(), cpu_out, 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestCambriconTranspose(flow.unittest.TestCase):
    def test_transpose(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_eager_transpose, _test_graph_transpose]
        arg_dict["device"] = ["mlu"]
        arg_dict["input_shape"] = [
            (10, 20, 20, 40),
            (256, 16, 32, 16),
        ]
        arg_dict["perm"] = [(2, 0, 1, 3), (1, 0, 2, 3), (3, 2, 1, 0), (3, 1, 2, 0)]
        arg_dict["dtype"] = [flow.float16, flow.float32]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
