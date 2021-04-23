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

import numpy as np
import oneflow as flow


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_ones(test_case):
        np_out = np.ones(10)
        of_out = flow.tmp.ones(10)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))
    
    def test_zeros(test_case):
        np_out2 = np.zeros((2, 3, 4, 5))
        of_out2 = flow.tmp.zeros((2, 3, 4, 5))
        test_case.assertTrue(np.allclose(of_out2.numpy(), np_out2))

    def test_arange(test_case):
        np_out = np.arange(5)
        of_out = flow.arange(0, 5)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

        np_out2 = np.arange(0, 20, 2)
        of_out2 = flow.arange(0, 20, 2)
        test_case.assertTrue(np.allclose(of_out2.numpy(), np_out2))



if __name__ == "__main__":
    unittest.main()
