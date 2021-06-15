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
import oneflow.experimental as flow


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestConsistentCastModule(flow.unittest.TestCase):
    def test_consistent_cast_module(test_case):
        relu = flow.nn.ReLU()

        consistent_relu = flow.consistent(
            (["S(0)"], ["S(0)"]),
            (
                [flow.placement("cpu", ["0:0"], None)],
                [flow.placement("cpu", ["0:0"], None)],
            ),
        )(relu)

        arr = np.random.randn(8, 16, 12, 5)
        np_out = np.maximum(0, arr)

        relu_mask = arr > 0
        out_diff = np.random.randn(8, 16, 12, 5)
        in_diff = out_diff * relu_mask

        x = flow.Tensor(arr, requires_grad=True)
        y = consistent_relu(x)
        y_diff = flow.Tensor(out_diff)
        y.backward(y_diff)
        test_case.assertTrue(np.allclose(y.numpy(), np_out, rtol=1e-05))
        test_case.assertTrue(np.allclose(x.grad.numpy(), in_diff, rtol=1e-05))

    def test_consistent_cast_function(test_case):
        relu = flow.nn.ReLU()

        def relu_func(x):
            return relu(x)

        arr = np.random.randn(8, 16, 12, 5)
        np_out = np.maximum(0, arr)

        relu_mask = arr > 0
        out_diff = np.random.randn(8, 16, 12, 5)
        in_diff = out_diff * relu_mask

        consisitent_relu_func = flow.consistent(
            (["S(0)"], ["S(0)"]),
            (
                [flow.placement("cpu", ["0:0"], None)],
                [flow.placement("cpu", ["0:0"], None)],
            ),
        )(relu_func)

        x = flow.Tensor(arr, requires_grad=True)
        y = consisitent_relu_func(x)
        y_diff = flow.Tensor(out_diff)
        y.backward(y_diff)
        test_case.assertTrue(np.allclose(y.numpy(), np_out, rtol=1e-05))
        test_case.assertTrue(np.allclose(x.grad.numpy(), in_diff, rtol=1e-05))

    def test_to_consistent(test_case):
        relu = flow.nn.ReLU()
        arr = np.random.randn(8, 16, 12, 5)
        np_out = np.maximum(0, arr)

        relu_mask = arr > 0
        out_diff = np.random.randn(8, 16, 12, 5)
        in_diff = out_diff * relu_mask

        consisitent_relu = relu.to_consistent(
            (["S(0)"], ["S(0)"]),
            (
                [flow.placement("cpu", ["0:0"], None)],
                [flow.placement("cpu", ["0:0"], None)],
            ),
        )

        x = flow.Tensor(arr, requires_grad=True)
        y = consisitent_relu(x)
        y_diff = flow.Tensor(out_diff)
        y.backward(y_diff)
        test_case.assertTrue(np.allclose(y.numpy(), np_out, rtol=1e-05))
        test_case.assertTrue(np.allclose(x.grad.numpy(), in_diff, rtol=1e-05))


if __name__ == "__main__":
    unittest.main()
