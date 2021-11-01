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
class TestArangeGraph(oneflow.unittest.TestCase):
    def test_arange_graph(test_case):
        of_eager_out = flow.arange(start=0, end=100, step=3, device=flow.device("cuda"))

        class ArangeGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self):
                return flow.arange(start=0, end=100, step=3, device=flow.device("cuda"))

        arange_g = ArangeGraph()
        of_lazy_out = arange_g()
        test_case.assertTrue(
            np.allclose(of_eager_out.numpy(), of_lazy_out.numpy(), 1e-05, 1e-05)
        )


if __name__ == "__main__":
    unittest.main()
