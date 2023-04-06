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

import numpy as np
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestParitalFC(flow.unittest.TestCase):
    @globaltest
    def test_parital_fc(test_case):
        placement = flow.placement.all("cuda")
        w = flow.randn(5000, 128, placement=placement, sbp=flow.sbp.split(0))
        label = flow.randint(
            0, 5000, (512,), placement=placement, sbp=flow.sbp.split(0)
        )
        num_sample = 500
        out = flow.distributed_partial_fc_sample(w, label, num_sample)
        test_case.assertTrue(out[0].shape == flow.Size([512]))
        test_case.assertTrue(out[1].shape == flow.Size([500]))
        test_case.assertTrue(out[2].shape == flow.Size([500, 128]))

        w = flow.randn(5000, 128, placement=placement, sbp=flow.sbp.broadcast)
        label = flow.randint(
            0, 5000, (512,), placement=placement, sbp=flow.sbp.split(0)
        )
        num_sample = 500
        out = flow.distributed_partial_fc_sample(w, label, num_sample)
        test_case.assertTrue(out[0].shape == flow.Size([512]))
        test_case.assertTrue(out[1].shape == flow.Size([500]))
        test_case.assertTrue(out[2].shape == flow.Size([500, 128]))


if __name__ == "__main__":
    unittest.main()
