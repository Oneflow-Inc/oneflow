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


class TestNLLLossError(flow.unittest.TestCase):
    def test_nllloss_reduction_type_error(test_case):
        with test_case.assertRaises(
            oneflow._oneflow_internal.exception.Exception
        ) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            target = flow.ones((4, 4), dtype=flow.float32)
            out = flow._C.nll_loss(x, target, None, 0, "just_test")

        test_case.assertTrue(
            'Check failed: reduction == "none" || reduction == "sum" || reduction == "mean" Reduction should be none, sum or mean.'
            in str(ctx.exception)
        )

    def test_nllloss_input_axis_error(test_case):
        with test_case.assertRaises(
            oneflow._oneflow_internal.exception.Exception
        ) as ctx:
            x = flow.ones((4, 1, 1, 1, 1, 1), dtype=flow.float32)
            target = flow.ones((4, 4), dtype=flow.float32)
            out = flow._C.nll_loss(x, target, None, 0, "none")

        test_case.assertTrue(
            "Check failed: (input_shape->NumAxes()) <= (5) (6 vs 5) The number of input's axis should be less equal to 5."
            in str(ctx.exception)
        )

    def test_nllloss_input_target_axis_error(test_case):
        with test_case.assertRaises(
            oneflow._oneflow_internal.exception.Exception
        ) as ctx:
            x = flow.ones((4, 1, 1), dtype=flow.float32)
            target = flow.ones((4, 4, 4), dtype=flow.float32)
            out = flow._C.nll_loss(x, target, None, 0, "none")

        test_case.assertTrue(
            "Check failed: (input_shape->NumAxes() - 1) == (target_shape->NumAxes()) (2 vs 3) The number of input's axis should be equal to the number of target's axis - 1."
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
