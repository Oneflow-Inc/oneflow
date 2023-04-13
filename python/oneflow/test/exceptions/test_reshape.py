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


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    def test_reshape_exception_invalid_dim(test_case):
        # torch exception and messge:
        #
        #   RuntimeError: Invalid shape dimension -2
        #
        x = flow.tensor((2, 2))
        with test_case.assertRaises(RuntimeError) as ctx:
            y = x.reshape((-2, 4))
        test_case.assertTrue("Invalid shape dimension -2" in str(ctx.exception))

    def test_reshape_exception_invalid_size(test_case):
        # torch exception and messge:
        #
        #   RuntimeError: shape '[2, 3, 5]' is invalid for input of size 24
        #
        x = flow.arange(24).reshape(2, 3, 4)
        with test_case.assertRaises(RuntimeError) as ctx:
            y = x.reshape((2, 3, 5))
        test_case.assertTrue("is invalid for input of size 24" in str(ctx.exception))

    def test_reshape_exception_only_one_dim_infered(test_case):
        # torch exception and messge:
        #
        #   RuntimeError: only one dimension can be inferred
        #
        x = flow.tensor((2, 2))
        with test_case.assertRaises(RuntimeError) as ctx:
            y = x.reshape((-1, -1))
        test_case.assertTrue("only one dimension can be inferred" in str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
