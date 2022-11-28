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
import re
import unittest

import oneflow as flow
import oneflow.unittest


class TestAutograd(flow.unittest.TestCase):
    def test_non_requires_grad_tensor_backward(test_case):
        x = flow.ones(4, 4)
        with test_case.assertRaises(Exception) as context:
            x.backward()
        test_case.assertIsNotNone(
            re.search(
                r"\nRuntimeError: element \d of tensors does not require grad and does not have a grad_fn",
                str(context.exception),
            )
        )


if __name__ == "__main__":
    unittest.main()
