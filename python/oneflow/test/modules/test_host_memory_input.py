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
from oneflow import nn
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestHostMemory(oneflow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_host_memory(test_case):
        x = flow.ones(2, 3, device="cuda")
        scalar = flow.Tensor([3.0], device="cuda")

        y = x + scalar
        out = y + scalar + y

        class HostMemoryInputGraph(nn.Graph):
            def __init__(self):
                super(HostMemoryInputGraph, self).__init__()

            def build(self, x, scalar):
                a = flow._C.host_scalar_add_by_tensor(x, scalar.cpu())
                b = flow._C.host_scalar_add_by_tensor(a, scalar)
                return a + b

        graph = HostMemoryInputGraph()
        lazy_out = graph(x, scalar)

        test_case.assertTrue(np.array_equal(out.numpy(), lazy_out.numpy()))

        a = flow._C.host_scalar_add_by_tensor(x, scalar.cpu())
        b = flow._C.host_scalar_add_by_tensor(a, scalar)
        eager_out = a + b
        test_case.assertTrue(np.array_equal(out.numpy(), eager_out.numpy()))

    @flow.unittest.skip_unless_1n2d()
    def test_host_memory_1n2d(test_case):
        x = flow.ones(
            2, 3, placement=flow.placement("cuda", [0, 1]), sbp=flow.sbp.broadcast
        )
        scalar = flow.Tensor(
            [3.0], placement=flow.placement("cuda", [0, 1]), sbp=flow.sbp.broadcast
        )

        y = x + scalar
        out = y + scalar + y

        class HostMemoryInputGraph(nn.Graph):
            def __init__(self):
                super(HostMemoryInputGraph, self).__init__()

            def build(self, x, scalar):
                a = flow._C.host_scalar_add_by_tensor(x, scalar.cpu())
                b = flow._C.host_scalar_add_by_tensor(a, scalar)
                return a + b

        graph = HostMemoryInputGraph()
        lazy_out = graph(x, scalar)

        test_case.assertTrue(np.array_equal(out.numpy(), lazy_out.numpy()))

        a = flow._C.host_scalar_add_by_tensor(x, scalar.cpu())
        b = flow._C.host_scalar_add_by_tensor(a, scalar)
        eager_out = a + b
        test_case.assertTrue(np.array_equal(out.numpy(), eager_out.numpy()))


if __name__ == "__main__":
    unittest.main()
