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
class TestTensorDetachGraph(oneflow.unittest.TestCase):
    def test_tensor_detach_graph(test_case):
        class TensorDetachGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, x):
                x += x
                y = x.detach()
                return x, y

        x = flow.randn(3, 4)
        res = TensorDetachGraph()(x)
        test_case.assertTrue(len(res) == 2)
        test_case.assertTrue(np.allclose(res[0], res[1], 1e-05, 1e-05))


if __name__ == "__main__":
    unittest.main()
