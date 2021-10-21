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
import oneflow as flow
from oneflow.nn.parallel import DistributedDataParallel as ddp
import oneflow.unittest

import numpy as np
import os


def np_allclose_with_shape(a, b, *args, **kwargs):
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, *args, **kwargs)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestDDP(flow.unittest.TestCase):
    def test_ddp_basic(test_case):
        class Mul(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = flow.nn.Parameter(flow.Tensor([1, 1]))

            def forward(self, x):
                return x * self.w

        rank = flow.env.get_rank()
        if rank == 0:
            x = flow.Tensor([1, 1])
        elif rank == 1:
            x = flow.Tensor([2, 2])
        else:
            raise ValueError()

        x = x.to("cuda")
        m = Mul().to("cuda")
        m = ddp(m)
        y = m(x)
        y.sum().backward()

        test_case.assertTrue(
            np_allclose_with_shape(m.w.grad.numpy(), np.array([1.5, 1.5]))
        )

    def test_ddp_with_unused_param(test_case):
        class Model(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = flow.nn.Parameter(flow.Tensor([1]))
                self.used_only_in_rank0 = flow.nn.Parameter(flow.Tensor([2]))
                self.unused_in_all_ranks = flow.nn.Parameter(flow.Tensor([3]))

            def forward(self, x):
                x = x * self.w
                if flow.env.get_rank() == 0:
                    x = x * self.used_only_in_rank0
                return x

        rank = flow.env.get_rank()
        if rank == 0:
            x = flow.Tensor([1])
        elif rank == 1:
            x = flow.Tensor([2])
        else:
            raise ValueError()

        x = x.to("cuda")
        m = Model().to("cuda")
        m = ddp(m)
        y = m(x)
        y.backward()

        test_case.assertTrue(np_allclose_with_shape(m.w.grad.numpy(), np.array([2])))
        test_case.assertTrue(
            np_allclose_with_shape(m.used_only_in_rank0.grad.numpy(), np.array([0.5]))
        )
        test_case.assertTrue(
            np_allclose_with_shape(m.unused_in_all_ranks.grad.numpy(), np.array([0]))
        )

    def test_out_of_order_execution(test_case):
        class Model(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = flow.nn.Parameter(flow.Tensor([1]))
                self.w2 = flow.nn.Parameter(flow.Tensor([2]))
                self.w3 = flow.nn.Parameter(flow.Tensor([3]))

            def forward(self, x):
                if flow.env.get_rank() == 0:
                    x *= self.w1
                    x *= self.w2
                    x *= self.w3
                else:
                    x *= self.w3
                    x *= self.w2
                    x *= self.w1
                return x

        rank = flow.env.get_rank()
        if rank == 0:
            x = flow.Tensor([1])
        elif rank == 1:
            x = flow.Tensor([2])
        else:
            raise ValueError()

        x = x.to("cuda")
        m = Model().to("cuda")
        m = ddp(m)
        y = m(x)
        y.backward()

        test_case.assertTrue(np_allclose_with_shape(m.w1.grad.numpy(), np.array([9])))
        test_case.assertTrue(np_allclose_with_shape(m.w2.grad.numpy(), np.array([4.5])))
        test_case.assertTrue(np_allclose_with_shape(m.w3.grad.numpy(), np.array([3])))

    def test_broadcast_buffer(test_case):
        rank = flow.env.get_rank()

        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", flow.tensor([1, 2]) * (rank + 1))

            def forward(self, x):
                res = self.buf + x
                self.buf.copy_(x)
                return res

        x = flow.tensor([2, 3]) * (rank + 1)
        x = x.to("cuda")

        m = CustomModule()
        m = m.to("cuda")
        m = ddp(m)

        y1 = m(x)
        y2 = m(x)

        m = CustomModule()
        m = m.to("cuda")
        m = ddp(m, broadcast_buffers=False)

        y3 = m(x)
        y4 = m(x)

        if rank == 0:
            test_case.assertTrue(np_allclose_with_shape(y1.numpy(), np.array([3, 5])))
            test_case.assertTrue(np_allclose_with_shape(y2.numpy(), np.array([4, 6])))
            test_case.assertTrue(np_allclose_with_shape(y3.numpy(), np.array([3, 5])))
            test_case.assertTrue(np_allclose_with_shape(y4.numpy(), np.array([4, 6])))
        elif rank == 1:
            test_case.assertTrue(np_allclose_with_shape(y1.numpy(), np.array([5, 8])))
            test_case.assertTrue(np_allclose_with_shape(y2.numpy(), np.array([6, 9])))
            test_case.assertTrue(np_allclose_with_shape(y3.numpy(), np.array([6, 10])))
            test_case.assertTrue(np_allclose_with_shape(y4.numpy(), np.array([8, 12])))
        else:
            raise ValueError()


if __name__ == "__main__":
    unittest.main()
