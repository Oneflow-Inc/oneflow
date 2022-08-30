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
import os
import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestSavedTensorHooks(flow.unittest.TestCase):
    def test_normal_saved_tensor_hooks(test_case):
        x = flow.ones(1, 2, 3).to("cuda").requires_grad_()
        y = flow.zeros(1, 2, 3).to("cuda").requires_grad_()
        tensor_list = []

        def pack(x):
            tensor_list.append(x)
            return len(tensor_list) - 1

        def unpack(x):
            return tensor_list[x]

        with flow.autograd.graph.saved_tensors_hooks(pack, unpack):
            z = x * y
        z.sum().backward()
        test_case.assertEqual(len(tensor_list), 2)
        test_case.assertTrue(np.array_equal(tensor_list[0], y))
        test_case.assertTrue(np.array_equal(tensor_list[1], x))
        test_case.assertTrue(np.allclose(x.grad, y))
        test_case.assertTrue(np.allclose(y.grad, x))

    def test_saved_tensor_hooks_in_autograd_function(test_case):
        x = flow.ones(1, 2, 3).to("cuda").requires_grad_()
        y = flow.zeros(1, 2, 3).to("cuda").requires_grad_()
        tensor_list = []

        def pack(x):
            tensor_list.append(x)
            return len(tensor_list) - 1

        def unpack(x):
            return tensor_list[x]

        class MulFunction(flow.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x * y

            @staticmethod
            def backward(ctx, dz):
                x, y = ctx.saved_tensors
                dx = dz * y
                dy = dz * x
                return dx, dy

        with flow.autograd.graph.saved_tensors_hooks(pack, unpack):
            z = MulFunction.apply(x, y)
        z.sum().backward()
        test_case.assertEqual(len(tensor_list), 2)
        test_case.assertTrue(np.array_equal(tensor_list[0], x))
        test_case.assertTrue(np.array_equal(tensor_list[1], y))
        test_case.assertTrue(np.allclose(x.grad, y))
        test_case.assertTrue(np.allclose(y.grad, x))


if __name__ == "__main__":
    unittest.main()
