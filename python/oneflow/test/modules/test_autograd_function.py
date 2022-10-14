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

import re

import unittest

import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow import autograd


class TestAutogradFunction(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_simple_input(test_case):
        class MyReLU(autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x.clamp(min=0.0, max=None)
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx, y_grad):
                x_grad = y_grad.clone()
                (x,) = ctx.saved_tensors
                x_grad[x < 0] = 0
                return x_grad

        np_arr = np.random.randn(4, 5)
        a = flow.tensor(np_arr).requires_grad_()
        # forward
        b = MyReLU.apply(a)
        test_case.assertTrue(np.allclose(b.numpy(), np_arr.clip(min=0.0)))
        # backward
        b.sum().backward()
        np_grad = np.ones((4, 5))
        np_grad[np_arr < 0] = 0.0
        test_case.assertTrue(np.allclose(a.grad.numpy(), np_grad))

    @flow.unittest.skip_unless_1n1d()
    def test_multi_input(test_case):
        class MyMatMul(autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                z = x * y
                ctx.save_for_backward(x, y)
                return z

            @staticmethod
            def backward(ctx, z_grad):
                x, y = ctx.saved_tensors
                x_grad = y * z_grad
                y_grad = x * z_grad
                return x_grad, y_grad

        np_arr0 = np.random.randn(4, 5)
        np_arr1 = np.random.randn(4, 5)
        a = flow.tensor(np_arr0).requires_grad_()
        b = flow.tensor(np_arr1).requires_grad_()
        # forward
        c = MyMatMul().apply(a, b)
        test_case.assertTrue(np.allclose(c.numpy(), np_arr0 * np_arr1))
        # backward
        c.sum().backward()
        test_case.assertTrue(np.allclose(a.grad.numpy(), np_arr1))
        test_case.assertTrue(np.allclose(b.grad.numpy(), np_arr0))

    @flow.unittest.skip_unless_1n1d()
    def test_non_differentiable_interface(test_case):
        class MyModule(autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                mul_res = x * y
                add_res = x + y
                ctx.save_for_backward(x, y)
                ctx.mark_non_differentiable(add_res)
                return mul_res, add_res

            @staticmethod
            def backward(ctx, mul_grad, add_grad=None):
                x, y = ctx.saved_tensors
                x_grad = y * mul_grad
                y_grad = x * mul_grad
                return x_grad, y_grad

        np_arr0 = np.random.randn(4, 5)
        np_arr1 = np.random.randn(4, 5)
        a = flow.tensor(np_arr0).requires_grad_()
        b = flow.tensor(np_arr1).requires_grad_()
        # forward
        c, d = MyModule().apply(a, b)
        test_case.assertTrue(np.allclose(c.numpy(), np_arr0 * np_arr1))
        test_case.assertFalse(d.requires_grad)
        test_case.assertTrue(d.grad_fn is None)
        # backward
        c.sum().backward()
        test_case.assertTrue(np.allclose(a.grad.numpy(), np_arr1))
        test_case.assertTrue(np.allclose(b.grad.numpy(), np_arr0))

    @flow.unittest.skip_unless_1n1d()
    def test_partial_inputs_requires_grad(test_case):
        class MyModule(autograd.Function):
            @staticmethod
            def forward(ctx, x, y, z):
                return x + y + z

            @staticmethod
            def backward(ctx, out_grad):
                return None, out_grad, None

        x = flow.randn(4, 5)
        y = flow.randn(4, 5).requires_grad_()
        z = flow.randn(4, 5)
        # forward
        res = MyModule.apply(x, y, z)
        test_case.assertTrue(
            np.allclose(res.numpy(), x.numpy() + y.numpy() + z.numpy())
        )
        # backward
        res.sum().backward()
        test_case.assertIsNone(x.grad)
        test_case.assertTrue(np.allclose(y.grad.numpy(), np.ones((4, 5))))
        test_case.assertIsNone(z.grad)

    @flow.unittest.skip_unless_1n1d()
    def test_dynamic_attr_for_ctx(test_case):
        class MyModule(autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.scale = 2.0
                return x * ctx.scale

            @staticmethod
            def backward(ctx, out_grad):
                return out_grad * ctx.scale

        x = flow.randn(4, 5).requires_grad_()
        # forward
        res = MyModule.apply(x)
        test_case.assertTrue(np.allclose(res.numpy(), x.numpy() * 2.0))
        # backward
        res.sum().backward()
        test_case.assertTrue(np.allclose(x.grad.numpy(), np.ones((4, 5)) * 2.0))

    @flow.unittest.skip_unless_1n1d()
    def test_backward_error_message(test_case):
        class MyModule(autograd.Function):
            @staticmethod
            def forward(ctx, x, y, z):
                return x + y + z

            @staticmethod
            def backward(ctx, out_grad):
                return None, out_grad

        x = flow.randn(4, 5)
        y = flow.randn(4, 5).requires_grad_()
        z = flow.randn(4, 5)
        res = MyModule.apply(x, y, z)
        with test_case.assertRaises(Exception) as exp:
            res.sum().backward()
        test_case.assertIsNotNone(
            re.search(
                r"RuntimeError: function MyModule returned an incorrect number of gradients \(expected \d, got \d\)",
                str(exp.exception),
            )
        )

    @flow.unittest.skip_unless_1n1d()
    def test_graph_test_multi_input(test_case):
        class MyMul(autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                z = x * y
                ctx.save_for_backward(x, y)
                return z

            @staticmethod
            def backward(ctx, z_grad):
                x, y = ctx.saved_tensors
                x_grad = 2 * y * z_grad
                y_grad = 3 * x * z_grad
                return x_grad, y_grad

        class MyAdd(autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return 2 * x + y

            @staticmethod
            def backward(ctx, z_grad):
                x_grad = z_grad
                y_grad = 2 * z_grad
                return x_grad, y_grad

        model = flow.nn.Linear(5, 4, bias=False)
        model.train()

        class MyGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.model = model
                optimizer = flow.optim.SGD(self.model.parameters())
                self.add_optimizer(optimizer)

            def build(self, x, y):
                x.retain_grad()
                y.retain_grad()
                self.model.weight.retain_grad()
                z = MyMul().apply(x, y)
                z = MyAdd().apply(z, self.model.weight)
                z.sum().backward()
                return z, x.grad, y.grad, self.model.weight.grad

        np_arr0 = np.random.randn(4, 5).astype(np.float32)
        np_arr1 = np.random.randn(4, 5).astype(np.float32)
        np_arr2 = np.random.randn(4, 5).astype(np.float32)
        a = flow.tensor(np_arr0).requires_grad_()
        b = flow.tensor(np_arr1).requires_grad_()
        model.weight.copy_(np_arr2)

        c, a_grad, b_grad, w_grad = MyGraph()(a, b)
        test_case.assertTrue(np.allclose(c.numpy(), 2 * np_arr0 * np_arr1 + np_arr2))
        test_case.assertTrue(np.allclose(a_grad.numpy(), 2 * np_arr1))
        test_case.assertTrue(np.allclose(b_grad.numpy(), 3 * np_arr0))
        test_case.assertTrue(np.allclose(w_grad.numpy(), 2 * np.ones_like(np_arr2)))

    @flow.unittest.skip_unless_1n1d()
    def test_autograd_function_memory(test_case):
        global_ctx = None

        class MyModule(autograd.Function):
            @staticmethod
            def forward(ctx, x):
                z = x.clone()
                ctx.save_for_backward(z)
                nonlocal global_ctx
                global_ctx = ctx
                return z

            @staticmethod
            def backward(ctx, out_grad):
                (x,) = ctx.saved_tensors
                return x

        x = flow.randn(5, 5).requires_grad_()
        res = MyModule.apply(x)
        test_case.assertTrue(global_ctx._is_data_valid())
        res.sum().backward()

        # ensure that global_ctx is released
        test_case.assertFalse(global_ctx._is_data_valid())


if __name__ == "__main__":
    unittest.main()
