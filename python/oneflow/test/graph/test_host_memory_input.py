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


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestReluGraph(oneflow.unittest.TestCase):
    def test_host_memory(test_case):
        x = flow.ones(2, 3, device="cuda")
        scalar = flow.Tensor([3.0], device="cuda")

        y = x + scalar
        z = y + scalar + y

        class HostMemoryInputGraph(nn.Graph):
            def __init__(self):
                super(HostMemoryInputGraph, self).__init__()

            def build(self, x, scalar):
                a = flow._C.host_scalar_add_by_tensor(x, scalar)
                b = flow._C.host_scalar_add_by_tensor(a, scalar)
                return a + b

        graph = HostMemoryInputGraph()
        w = graph(x, scalar)

        test_case.assertTrue(np.array_equal(z.numpy(), w.numpy()))


if __name__ == "__main__":
    unittest.main()
