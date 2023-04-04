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

    @autotest(n=1, auto_backward=False, check_graph=False)
    def test_out_grad_with_different_dtype(test_case):
        x = random_tensor(ndim=2, requires_grad=True)
        y = x.sum()
        y.backward(torch.tensor(False))
        return x.grad

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

    def test_autograd_set_acc_grad_and_backward(test_case):
        for _ in range(5):
            ndim = 2
            dims = [random(1, 5).to(int).value() for _ in range(ndim)]
            x = torch.randn(*dims).requires_grad_()
            np_arr = np.random.rand(*dims)
            init_grad = torch.tensor(np_arr).to(x.dtype)
            x.pytorch.grad = init_grad.pytorch
            x.oneflow.grad = init_grad.oneflow

            x.sum().backward()
            test_case.assertTrue(
                np.allclose(
                    x.grad.oneflow.numpy(), x.grad.pytorch.cpu().detach().numpy()
                )
            )

    @autotest(n=1, check_graph=False)
    def test_requires_grad_tensor_inplace_and_backward(test_case):
        random_shape = [random(1, 10).to(int) for _ in range(4)]
        x = random_tensor(4, *random_shape, requires_grad=False)
        y = random_tensor(4, *random_shape, requires_grad=True)
        x += y
        return x

    @autotest(n=1, check_graph=False)
    def test_retain_grad_for_leaf_tensor(test_case):
        random_shape = [random(1, 10).to(int) for _ in range(4)]
        x = random_tensor(4, *random_shape, requires_grad=True)
        y = x * 2
        x.retain_grad()
        return y

    @autotest(n=1, auto_backward=False, check_graph=False)
    def test_run_backward_and_grad_for_same_tensor(test_case):
        random_shape = [random(1, 10).to(int) for _ in range(4)]
        x = random_tensor(4, *random_shape, requires_grad=True)
        y = x ** 2
        y.sum().backward()
        test_case.assertTrue(
            np.allclose(x.grad.oneflow.numpy(), x.grad.pytorch.numpy())
        )

        y = x ** 2
        x_grad = torch.autograd.grad(y.sum(), x)[0]
        test_case.assertTrue(
            np.allclose(x_grad.oneflow.numpy(), x_grad.pytorch.numpy())
        )
        test_case.assertTrue(
            np.allclose(x.grad.oneflow.numpy(), x_grad.oneflow.numpy())
        )

    @autotest(n=1, auto_backward=False, check_graph=False)
    def test_no_grad_domain_call_backward(test_case):
        random_shape = [random(1, 10).to(int).value() for _ in range(4)]
        with flow.no_grad():
            x = flow.rand(*random_shape).requires_grad_()
            with flow.enable_grad():
                y = x * 2
            flow.autograd.backward(y, flow.ones_like(y))
        test_case.assertTrue(np.array_equal(x.grad.numpy(), np.full(random_shape, 2.0)))

    @autotest(n=1, auto_backward=False, check_graph=False)
    def test_acc_grad_inplace_update(test_case):
        random_shape = [random(1, 5).to(int).value() for _ in range(4)]
        x = flow.rand(*random_shape).requires_grad_()
        y = flow.rand(*random_shape).requires_grad_()

        z = x / (x + y)
        z.sum().backward()
        id_x_grad = id(x.grad)
        id_y_grad = id(y.grad)

        z = x / (x + y)
        z.sum().backward()
        test_case.assertEqual(id_x_grad, id(x.grad))
        test_case.assertEqual(id_y_grad, id(y.grad))


if __name__ == "__main__":
    unittest.main()
