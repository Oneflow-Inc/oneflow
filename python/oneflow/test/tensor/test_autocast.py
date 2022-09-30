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
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "skip test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestAutoCast(flow.unittest.TestCase):
    @autotest(n=1, auto_backward=True, check_graph=False)
    def test_autocast_half_mm(test_case):
        a = random_tensor(2, 2, 3).to("cuda")
        b = random_tensor(2, 3, 4).to("cuda")
        with torch.autocast("cuda"):
            x = torch.mm(a, b)
        return x

    @autotest(n=1, auto_backward=True, check_graph=False)
    def test_autocast_half_mm_add(test_case):
        a = random_tensor(2, 2, 3).to("cuda")
        b = random_tensor(2, 3, 4).to("cuda")
        c = random_tensor(2, 2, 4).to("cuda")
        with torch.autocast("cuda"):
            x = torch.mm(a, b)
            y = x + c
        return x.float() + y.float()

    def test_autocast_graph(test_case):
        class LinearGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear = flow.nn.Linear(3, 4, bias=False).cuda().half()

            def build(self, x):
                return self.linear(x)

        x = flow.Tensor(3, 3).cuda()

        with flow.autocast(device_type="cuda"):
            linear = LinearGraph()
            y = linear(x)
            test_case.assertTrue(y.dtype == flow.float16)


if __name__ == "__main__":
    unittest.main()
