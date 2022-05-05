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


class TestTripletMarginLossError(flow.unittest.TestCase):
    def test_triplet_margin_loss_reduce_type_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            anchor = flow.ones((3, 3), dtype=flow.float32)
            positive = flow.ones((3, 3), dtype=flow.float32)
            negative = flow.ones((3, 3), dtype=flow.float32)

            triplet_loss = flow._C.triplet_margin_loss(
                anchor,
                positive,
                negative,
                margin=0.001,
                p=2,
                eps=1e-5,
                swap=False,
                reduction="just_test",
            )

        test_case.assertTrue(
            'Check failed: [&]() -> bool { if ((reduction != "none") && (reduction != "sum") && (reduction != "mean")) return false; return true; }() Reduction should be none, sum or mean.'
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
