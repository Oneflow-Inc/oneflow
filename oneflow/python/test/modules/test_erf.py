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
import oneflow.experimental as flow
from scipy import special


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestNegativeModule(flow.unittest.TestCase):
    def test_erf(test_case):
        input = flow.Tensor(
            np.array([1.0, -1.3, 2.3]).astype(np.float32), dtype=flow.float32
        )
        of_out = flow.erf(input)
        np_out = special.erf(input.numpy())
        test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

    def test_tensor_erf(test_case):
        input = flow.Tensor(
            np.array([1.0, -1.3, 2.3]).astype(np.float32), dtype=flow.float32
        )
        of_out = input.erf()
        np_out = special.erf(input.numpy())
        test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


if __name__ == "__main__":
    unittest.main()