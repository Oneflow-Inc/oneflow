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
from collections import OrderedDict

import os
import numpy as np
import time
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


class TestActivationError(flow.unittest.TestCase):
    def test_relu_inplace_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            x.relu_()
        test_case.assertTrue(
            "a leaf Tensor that requires grad is being used in an in-place operation"
            in str(context.exception)
        )

    def test_prelu_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            m = flow.nn.PReLU(5)
            y = m(x)
        test_case.assertTrue(
            "num_parameters in prelu must be 1 or 4" in str(context.exception)
        )

    def test_celu_inplace_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            m = flow.nn.CELU(alpha=1.0, inplace=True)
            y = m(x)
        test_case.assertTrue(
            "a leaf Tensor that requires grad is being used in an in-place operation"
            in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
