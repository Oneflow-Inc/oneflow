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


class TestCTCLossError(flow.unittest.TestCase):
    def test_ctcloss_reduction_type_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((5, 2, 3), dtype=flow.float32)
            targets = flow.tensor([[1, 2, 2], [1, 2, 2]], dtype=flow.int32)
            input_lengths = flow.tensor([5, 5], dtype=flow.int32)
            target_lengths = flow.tensor([3, 3], dtype=flow.int32)
            max_target_length = 0
            if targets.ndim == 1:
                max_target_length = target_lengths.max().item()
            elif targets.ndim == 2:
                max_target_length = targets.shape[1]
            loss = flow._C.ctc_loss(
                x,
                targets,
                input_lengths,
                target_lengths,
                max_target_length,
                blank=0,
                zero_infinity=False,
                reduction="just_test",
            )
        test_case.assertTrue(
            'Check failed: [&]() -> bool { if ((reduction != "none") && (reduction != "sum") && (reduction != "mean")) return false; return true; }() Reduction should be none, sum or mean.'
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
