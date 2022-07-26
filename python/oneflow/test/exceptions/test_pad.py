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
import oneflow.nn.functional as F
import torch


@flow.unittest.skip_unless_1n1d()
class TestPad(flow.unittest.TestCase):
    def test_torch_type(test_case):
        with test_case.assertRaises(TypeError) as exp:
            F.pad(torch.randn(2, 2))
        test_case.assertTrue(
            "pad() missing 1 required positional argument: 'pad'" in str(exp.exception)
        )

    def test_numpy_type(test_case):
        import numpy as np

        with test_case.assertRaises(TypeError) as exp:
            F.pad(np.random.randn(2, 2))
        test_case.assertTrue(
            "pad() missing 1 required positional argument: 'pad'" in str(exp.exception)
        )


if __name__ == "__main__":
    unittest.main()
