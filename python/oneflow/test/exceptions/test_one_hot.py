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


class TestOnehotError(flow.unittest.TestCase):
    def test_onehot_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((3, 3), dtype=flow.float32)
            out = flow._C.one_hot(x, 3, 0.9, 0)
        test_case.assertTrue(
            "RuntimeError : one_hot is only applicable to index tensor."
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
