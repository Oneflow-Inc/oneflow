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

import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow import autograd


def _compare_model_between_graph_and_eager(autograd_func, inputs):
    linear_init_data = np.random.randn(5, 5).astype("float32")

    class EagerModel(flow.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = flow.nn.Linear(5, 5, bias=False)
            self.linear.weight.data = flow.tensor(linear_init_data)

        def forward(self, *args):
            args = [self.linear(x) for x in args]
            y = autograd_func.apply(*args)
            return y

    class Graph(flow.nn.Graph):
        def __init__(self) -> None:
            super().__init__()
            self.model = EagerModel()
            optimizer = flow.optim.SGD(self.model.parameters(), lr=1)
            self.add_optimizer(optimizer)

        def build(self, *args):
            result = self.model(*args)
            # if isinstance(result, (tuple, list)):
            #     result = flow.stack(result)
            loss = result.sum()
            loss.backward()
            return loss

    # import ipdb; ipdb.set_trace()
    graph = Graph()
    inp = [flow.tensor(v).requires_grad_() for v in inputs]
    graph_loss = graph(*inp)

    inp = [flow.tensor(v).requires_grad_() for v in inputs]
    model = EagerModel()
    optim = flow.optim.SGD(model.parameters(), lr=1)
    result = model(*inp)
    # if isinstance(result, (tuple, list)):
    #     result = flow.stack(result)
    eager_loss = result.sum()
    optim.zero_grad()
    eager_loss.backward()
    optim.step()
    return graph.state_dict()["model"]["linear.weight"], model.linear.weight


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


class TestGraphAutogradFunction(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_simple_input(test_case):
        class SquareFunc(flow.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                ctx.save_for_backward(inp)
                return inp * inp

            @staticmethod
            def backward(ctx, grad):
                (inp,) = ctx.saved_tensors
                return inp * grad

        inputs = [np.random.randn(5, 5).astype("float32")]
        graph_weight, eager_weight = _compare_model_between_graph_and_eager(
            SquareFunc, inputs
        )
        test_case.assertTrue(np.allclose(graph_weight.numpy(), eager_weight.numpy()))

    @flow.unittest.skip_unless_1n2d()
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

        inputs = [np.random.randn(5, 5).astype("float32") for i in range(2)]
        graph_weight, eager_weight = _compare_model_between_graph_and_eager(
            MyMatMul, inputs
        )
        test_case.assertTrue(np.allclose(graph_weight.numpy(), eager_weight.numpy()))

    @flow.unittest.skip_unless_1n2d()
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

        inputs = [np.random.randn(5, 5).astype("float32") for i in range(2)]
        graph_weight, eager_weight = _compare_model_between_graph_and_eager(
            MyModule, inputs
        )
        test_case.assertTrue(np.allclose(graph_weight.numpy(), eager_weight.numpy()))


if __name__ == "__main__":
    unittest.main()
