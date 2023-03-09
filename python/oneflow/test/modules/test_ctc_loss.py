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

import os
import unittest
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestCTCLoss1n1d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    # This test case can always success out of ci container, but will get error in ci container for unknown reason: error:
    # 'oneflow.ctc_loss' op attribute 'blank' failed to satisfy constraint: 32-bit signed integer attribute
    # loc("-":0:0): error: Failed to run round-trip passes
    @autotest(n=5, check_graph=False)
    def test_ctc_loss_with_diff_device_input(test_case):
        log_probs = torch.tensor(
            [
                [[-1.1031, -0.7998, -1.5200], [-0.9808, -1.1363, -1.1908]],
                [[-1.2258, -1.0665, -1.0153], [-1.1135, -1.2331, -0.9671]],
                [[-1.3348, -0.6611, -1.5118], [-0.9823, -1.2355, -1.0941]],
                [[-1.3850, -1.3273, -0.7247], [-0.8235, -1.4783, -1.0994]],
                [[-0.9049, -0.8867, -1.6962], [-1.4938, -1.3630, -0.6547]],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32, device="cuda")
        input_lengths = torch.tensor([5, 5], dtype=torch.int32)
        target_lengths = torch.tensor([3, 3], dtype=torch.int32)
        loss_mean = torch.nn.CTCLoss(reduction=oneof("mean", "none", "sum", nothing()))
        out = loss_mean(log_probs, targets, input_lengths, target_lengths)
        return out

    @autotest(n=5, check_graph=False)
    def test_ctc_loss_functional(test_case):
        device_random = random_device()
        log_probs = random_tensor(ndim=3, dim0=5, dim1=2, dim2=3).to(device_random)
        targets = random_tensor(ndim=2, dim0=2, dim1=3, low=1, high=3, dtype=int).to(
            device_random
        )
        input_lengths = torch.tensor([5, 5], dtype=torch.int32)
        target_lengths = torch.tensor([3, 3], dtype=torch.int32)
        out = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction=oneof("mean", "none", "sum", nothing()),
        )
        return out


if __name__ == "__main__":
    unittest.main()
