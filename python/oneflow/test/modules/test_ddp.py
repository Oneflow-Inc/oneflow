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

# Test import from oneflow.nn.parallel.distributed
from oneflow.nn.parallel.distributed import DistributedDataParallel
from oneflow.nn.parallel import DistributedDataParallel as ddp
from oneflow.utils.global_view import global_mode, to_global
import oneflow.unittest

import numpy as np
import os


def np_allclose_with_shape(a, b, *args, **kwargs):
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, *args, **kwargs)


test_device = ["cpu"] if os.getenv("ONEFLOW_TEST_CPU_ONLY") else ["cpu", "cuda"]
placements = [flow.placement("cpu", ranks=[0, 1]), flow.placement("cuda", ranks=[0, 1])]
sbps = [flow.sbp.broadcast, flow.sbp.split(0)]


@flow.unittest.skip_unless_1n2d()
class TestDDP(flow.unittest.TestCase):
    def _test_ddp_basic(test_case, dev_type):
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

        x = x.to(dev_type)
        m = Mul().to(dev_type)
        m = ddp(m)
        y = m(x)
        y.sum().backward()

        test_case.assertTrue(
            np_allclose_with_shape(m.w.grad.numpy(), np.array([1.5, 1.5]))
        )

    def test_ddp_basic(test_case):
        for dev_type in test_device:
            test_case._test_ddp_basic(dev_type)

    def _test_ddp_with_global_mode(test_case, placement, sbp):
        class ParamModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = flow.ones([2], requires_grad=True).to_global(
                    placement, sbp
                )
                
            def forward(self, x):
                return x * self.w
            
        class MulGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.param = ParamModule()
                of_sgd = flow.optim.SGD(
                    [{"params": self.param.parameters()}], lr=0.001, momentum=0.9,
                )
                self.add_optimizer(of_sgd)

            def build(self, x):
                with global_mode(True, placement, sbp):
                    device = self.param.w.device
                    x = x.to(device)
                    out = self.param(x)

                loss = out.sum()
                loss.backward()

                return out

        rank = flow.env.get_rank()
        if rank == 0:
            with global_mode(True, placement, sbp):
                x = flow.ones([2])
        elif rank == 1:
            with global_mode(True, placement, sbp):
                x = flow.ones([2]) * 2
        else:
            raise ValueError()

        m = MulGraph()
        y = m(x)

        test_case.assertTrue(
            np_allclose_with_shape(m.param.w.grad.numpy(), np.array([1.5, 1.5]))
        )

    def test_ddp_with_global_mode(test_case):
        for placement in placements:
            for sbp in sbps:
                test_case._test_ddp_with_global_mode(placement, sbp)

    def _test_ddp_multiple_buckets(test_case, dev_type):
        class Mul(flow.nn.Module):
            def __init__(self):
                super().__init__()
                for i in range(10):
                    self.register_parameter(
                        f"w{i}", flow.nn.Parameter(flow.Tensor([i % 2 + 1, i % 2 + 1]))
                    )

            def forward(self, x):
                for i in range(10):
                    x = x * getattr(self, f"w{i}")
                return x

        rank = flow.env.get_rank()
        if rank == 0:
            x = flow.Tensor([1, 1])
        elif rank == 1:
            x = flow.Tensor([2, 2])
        else:
            raise ValueError()

        x = x.to(dev_type)
        m = Mul().to(dev_type)
        m = ddp(m, bucket_size=3)

        y = m(x)
        y.sum().backward()

        for i in range(10):
            test_case.assertTrue(
                np_allclose_with_shape(
                    getattr(m, f"w{i}").grad.numpy(),
                    np.array([48, 48]) if i % 2 == 0 else np.array([24, 24]),
                )
            )

    def test_ddp_multiple_buckets(test_case):
        for dev_type in test_device:
            test_case._test_ddp_multiple_buckets(dev_type)

    def _test_ddp_with_unused_param(test_case, dev_type):
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

        x = x.to(dev_type)
        m = Model().to(dev_type)
        m = ddp(m, bucket_size=2)
        y = m(x)
        y.backward()

        test_case.assertTrue(np_allclose_with_shape(m.w.grad.numpy(), np.array([2])))
        test_case.assertTrue(
            np_allclose_with_shape(m.used_only_in_rank0.grad.numpy(), np.array([0.5]))
        )
        test_case.assertTrue(
            np_allclose_with_shape(m.unused_in_all_ranks.grad.numpy(), np.array([0]))
        )

    def test_ddp_with_unused_param(test_case):
        for dev_type in test_device:
            test_case._test_ddp_with_unused_param(dev_type)

    def _test_out_of_order_execution(test_case, dev_type):
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

        x = x.to(dev_type)
        m = Model().to(dev_type)
        m = ddp(m, bucket_size=1)
        y = m(x)
        y.backward()

        test_case.assertTrue(np_allclose_with_shape(m.w1.grad.numpy(), np.array([9])))
        test_case.assertTrue(np_allclose_with_shape(m.w2.grad.numpy(), np.array([4.5])))
        test_case.assertTrue(np_allclose_with_shape(m.w3.grad.numpy(), np.array([3])))

    def test_out_of_order_execution(test_case):
        for dev_type in test_device:
            test_case._test_out_of_order_execution(dev_type)

    def _test_ddp_with_partial_requires_grad_parameter(test_case, dev_type):
        class Model(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = flow.nn.Parameter(flow.Tensor([1]), requires_grad=False)
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

        x = x.to(dev_type)
        m = Model().to(dev_type)
        m = ddp(m, bucket_size=1)
        y = m(x)
        y.backward()

        test_case.assertTrue(np_allclose_with_shape(m.w2.grad.numpy(), np.array([4.5])))
        test_case.assertTrue(np_allclose_with_shape(m.w3.grad.numpy(), np.array([3])))

    def test_ddp_with_partial_requires_grad_parameter(test_case):
        for dev_type in test_device:
            test_case._test_ddp_with_partial_requires_grad_parameter(dev_type)

    def _test_ddp_two_iters(test_case, dev_type):
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

        x = x.to(dev_type)
        m = Mul().to(dev_type)
        m = ddp(m)

        for _ in range(2):
            y = m(x)
            y.sum().backward()

        test_case.assertTrue(np_allclose_with_shape(m.w.grad.numpy(), np.array([3, 3])))

    def test_ddp_two_iters(test_case):
        for dev_type in test_device:
            test_case._test_ddp_two_iters(dev_type)

    def _test_broadcast_buffer(test_case, dev_type):
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
        x = x.to(dev_type)

        m = CustomModule()
        m = m.to(dev_type)
        m = ddp(m)

        y1 = m(x)
        y2 = m(x)

        m = CustomModule()
        m = m.to(dev_type)
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

    def test_broadcast_buffer(test_case):
        for dev_type in test_device:
            test_case._test_broadcast_buffer(dev_type)


if __name__ == "__main__":
    unittest.main()
