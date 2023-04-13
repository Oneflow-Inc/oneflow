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
import os

import oneflow as flow
import oneflow.unittest
import numpy as np


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestCheckpointing(flow.unittest.TestCase):
    def test_checkpointing(test_case):
        relu_forward_num = 0
        relu_backward_num = 0

        class MyReLU(flow.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                nonlocal relu_forward_num
                relu_forward_num += 1
                y = flow.relu(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, dy):
                nonlocal relu_backward_num
                relu_backward_num += 1
                y = ctx.saved_tensors[0]
                return dy * (y > 0)

        class M(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = flow.nn.Conv2d(3, 3, 3)
                self.conv2 = flow.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                x = self.conv1(x)
                if checkpointing:
                    x = flow.utils.checkpoint.checkpoint(MyReLU.apply, x)
                else:
                    x = MyReLU.apply(x)
                x = self.conv2(x)
                return x

        x1 = flow.randn(1, 3, 8, 16).requires_grad_()
        x2 = x1.detach().clone().requires_grad_()

        m = M()

        checkpointing = True
        y1 = m(x1)
        y1.sum().backward()

        checkpointing = False
        y2 = m(x2)
        y2.sum().backward()

        test_case.assertTrue(np.array_equal(y1, y2))
        test_case.assertTrue(np.array_equal(x1.grad, x2.grad))
        test_case.assertEqual(relu_forward_num, 3)
        test_case.assertEqual(relu_backward_num, 2)


if __name__ == "__main__":
    unittest.main()
