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
import oneflow.unittest
import oneflow as flow
import numpy as np


class TestInTopK(flow.unittest.TestCase):
    def test_in_top_k_error_msg(test_case):
        arr = np.array([1, 1])
        targets = flow.Tensor(arr)
        targets = flow.cast(targets, flow.float)
        arr = np.array([[0.8, 0.6, 0.3], [0.1, 0.6, 0.4]])
        predictions = flow.Tensor(arr)
        with test_case.assertRaises(RuntimeError) as ctx:
            flow.in_top_k(targets, predictions, 1)
        test_case.assertTrue(
            "targets data type must be index type" in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
