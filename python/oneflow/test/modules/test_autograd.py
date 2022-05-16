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
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

from oneflow.test_utils.test_util import GenArgList


def _test_autograd_backward(test_case, shape, device):
    np_input = np.random.rand(*shape)
    of_input = flow.tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = of_input ** 2
    of_out_sum = of_out.sum()
    of_out_sum.backward()
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), np_input * 2, 0.0001, 0.0001)
    )
    of_input = flow.tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = of_input ** 2
    of_out_sum = of_out.sum()
    of_out_sum.backward(flow.ones_like(of_out_sum) * 3)
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), np_input * 6, 0.0001, 0.0001)
    )
    of_input = flow.tensor(
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
    of_input = flow.tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = of_input ** 2
    of_out_sum = of_out.sum()
    grad = flow.autograd.grad(of_out_sum, of_input)[0]
    test_case.assertTrue(of_input.grad is None)
    test_case.assertTrue(np.allclose(grad.numpy(), np_input * 2, 0.0001, 0.0001))
    of_input = flow.tensor(
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

    @autotest(n=10, auto_backward=True, rtol=1e-3, atol=1e-3, check_graph=True)
    def test_accumulate_grad(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        y = random_tensor(ndim=ndim, requires_grad=True).to(device)
        return x / (x + y)

    @autotest(n=10, auto_backward=True, rtol=1e-3, atol=1e-3, check_graph=True)
    def test_0dim_accumulate_grad(test_case):
        device = random_device()
        ndim = 0
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        y = random_tensor(ndim=ndim, requires_grad=True).to(device)
        return x / (x + y)

    @autotest(n=10, auto_backward=True, rtol=1e-3, atol=1e-3, check_graph=True)
    def test_scalar_leaf_tensor_backward(test_case):
        device = random_device()
        ndim = 0
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        return x

    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_grad_grad(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        y = x * x * x
        x_grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]
        x_grad_grad = torch.autograd.grad(
            outputs=x_grad, inputs=x, grad_outputs=torch.ones_like(x_grad)
        )[0]
        return x_grad_grad

    @autotest(n=10, auto_backward=False, rtol=1e-3, atol=1e-3, check_graph=False)
    def test_autograd_multiple_times(test_case):
        device = random_device()
        ndim = random(1, 4).to(int).value()
        dims = [random(0, 10).to(int) for _ in range(ndim)]
        x = random_tensor(ndim, *dims, requires_grad=True)
        x1 = x.to(device)
        y = random_tensor(ndim, *dims, requires_grad=True)
        y1 = y.to(device)
        z = x1 + y1

        for _ in range(10):
            z.sum().backward()
        return (x.grad, y.grad)


if __name__ == "__main__":
    unittest.main()
