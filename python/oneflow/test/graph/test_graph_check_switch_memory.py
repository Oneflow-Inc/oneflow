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

class OnesModule(flow.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self):
        ones = flow.ones(*self.shape)
        return ones

class RandomModule(flow.nn.Module):
    def __init__(self, shape):
        super().__init__()    
        self.shape = shape

    def forward(self):
        random = flow.randn(*self.shape)
        return random

def _test_memmory_graph(test_case, device, ):

    shape = (100, 3, 512, 512)

    ones_np = np.ones(shape)
    
    random = RandomModule(shape).to(flow.device("cuda"))
    of_eager_out = random()

    Ones = OnesModule(shape).to(flow.device("cuda"))

    class OnesGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.ones = Ones

        def build(self):
            return self.ones()

    ones_g = OnesGraph()

    for i in range(1000):
        of_eager_out = random()
        of_lazy_out = ones_g()
        test_case.assertTrue(np.array_equal(of_lazy_out.numpy(), ones_np))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestLinearGraph(oneflow.unittest.TestCase):
    def test_linear_graph_gpu(test_case):
        _test_memmory_graph(test_case, flow.device("cuda"))

    def test_linear_graph_cpu(test_case):
        _test_memmory_graph(test_case, flow.device("cpu"))


if __name__ == "__main__":
    unittest.main()
