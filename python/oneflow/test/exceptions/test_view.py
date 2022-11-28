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
    def test_view_exception(test_case):
        # torch exception and messge:
        #
        #   RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        #
        a = flow.arange(9).reshape(3, 3)
        b = a.permute(1, 0)
        with test_case.assertRaises(RuntimeError) as ctx:
            print(b.view(9))
        test_case.assertTrue(
            "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead."
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
