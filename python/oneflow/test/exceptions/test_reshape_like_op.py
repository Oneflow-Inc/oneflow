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


class TestReshapeLikeOp(flow.unittest.TestCase):
    def test_reshape_like_size_match_err(test_case):
        a = flow.tensor([1, 1])
        b = flow.tensor([[1, 1, 1], [1, 1, 1]])
        with test_case.assertRaises(RuntimeError) as ctx:
            flow._C.reshape_like(a, b)
        test_case.assertTrue(
            "The element number of the in tensor must be equal to the element number of the like tensor"
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
