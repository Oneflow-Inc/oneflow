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


def _test_narrow_forward(test_case, shape, dim, start_length, device, dtype):
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device), dtype=dtype)

    mlu_out = flow.narrow(x, dim=dim, start=start_length[0], length=start_length[1])
    cpu_out = flow.narrow(
        x.cpu(), dim=dim, start=start_length[0], length=start_length[1]
    )
    test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out.numpy(), 0.0001, 0.0001))

    class NarrowGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, x):
            return flow.narrow(
                x, dim=dim, start=start_length[0], length=start_length[1]
            )

    graph = NarrowGraph()
    graph_out = graph(x)
    test_case.assertTrue(
        np.allclose(graph_out.numpy(), cpu_out.numpy(), 0.0001, 0.0001)
    )


def _test_narrow_backward(test_case, shape, dim, start_length, device, dtype):
    np_arr = np.random.randn(*shape)
    x = flow.tensor(np_arr, device=flow.device(device), dtype=dtype, requires_grad=True)
    x_cpu = flow.tensor(np_arr, device="cpu", dtype=dtype, requires_grad=True)

    mlu_out = flow.narrow(x, dim=dim, start=start_length[0], length=start_length[1])
    cpu_out = flow.narrow(
        x_cpu, dim=dim, start=start_length[0], length=start_length[1]
    )
    test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out.numpy(), 0.0001, 0.0001))

    np_grad = np.random.randn(*mlu_out.shape)
    mlu_out.backward(flow.tensor(np_grad, device=flow.device(device), dtype=dtype))
    cpu_out.backward(flow.tensor(np_grad, device="cpu", dtype=dtype))

    test_case.assertTrue(np.allclose(x.grad.numpy(), x_cpu.grad.numpy(), 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestNarrowCambriconModule(flow.unittest.TestCase):
    def test_narrow(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_narrow_forward,
        ]
        arg_dict["shape"] = [(3, 4, 5), (6, 7, 8)]
        arg_dict["dim"] = [0, 1, 2]
        arg_dict["start_length"] = [(0, 2), (2, 1), (1, 2)]
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

    def test_narrow_backward(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_narrow_backward
        ]
        arg_dict["shape"] = [(3, 4, 5), (6, 7, 8)]
        arg_dict["dim"] = [0, 1, 2]
        arg_dict["start_length"] = [(0, 2), (2, 1), (1, 2)]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
