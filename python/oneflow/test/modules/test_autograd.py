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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_autograd_backward(test_case, shape, device):
    np_input = np.random.rand(*shape)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = of_input ** 2
    of_out_sum = of_out.sum()
    of_out_sum.backward()
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), np_input * 2, 0.0001, 0.0001)
    )
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = of_input ** 2
    of_out_sum = of_out.sum()
    of_out_sum.backward(flow.ones_like(of_out_sum) * 3)
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), np_input * 6, 0.0001, 0.0001)
    )
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = of_input ** 2
    of_out_sum = of_out.sum()
    of_out_sum.backward(retain_graph=True)
    of_out_sum.backward(retain_graph=True)
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), np_input * 4, 0.0001, 0.0001)
    )


def _test_autograd_grad(test_case, shape, device):
    np_input = np.random.rand(*shape)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = of_input ** 2
    of_out_sum = of_out.sum()
    grad = flow.autograd.grad(of_out_sum, of_input)[0]
    test_case.assertTrue(of_input.grad is None)
    test_case.assertTrue(np.allclose(grad.numpy(), np_input * 2, 0.0001, 0.0001))
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = of_input ** 2
    of_out_sum = of_out.sum()
    grad = flow.autograd.grad(of_out_sum, of_input, flow.ones_like(of_out_sum) * 3)[0]
    test_case.assertTrue(np.allclose(grad.numpy(), np_input * 6, 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestAutograd(flow.unittest.TestCase):
    def test_autograd_interface(test_case):
        arg_dict = OrderedDict()
        arg_dict["case"] = [_test_autograd_backward, _test_autograd_grad]
        arg_dict["shape"] = [(2, 3), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
