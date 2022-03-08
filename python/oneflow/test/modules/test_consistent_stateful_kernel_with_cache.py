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


def _test_global_stateful_kernel_with_inpersistent_state(test_case, placement, sbp):
    x = (
        flow.arange(64)
        .reshape(8, 8)
        .to_global(flow.env.all_device_placement("cpu"), flow.sbp.broadcast)
    )
    x = x.to_global(placement, sbp)
    y = flow._C.logical_slice(x, [0, 0], [3, 1], [1, 1])
    y_np = np.array([[0], [8], [16]])
    test_case.assertTrue(
        np.array_equal(
            y.to_global(flow.env.all_device_placement("cpu"), flow.sbp.broadcast)
            .to_local()
            .numpy(),
            y_np,
        )
    )
    x = x.to_global(sbp=flow.sbp.split(1))
    y = flow._C.logical_slice(x, [0, 0], [3, 1], [1, 1])
    test_case.assertTrue(
        np.array_equal(
            y.to_global(flow.env.all_device_placement("cpu"), flow.sbp.broadcast)
            .to_local()
            .numpy(),
            y_np,
        )
    )


class TestStatefulKernelWithInpersistentState(flow.unittest.TestCase):
    @globaltest
    def test_global_stateful_kernel_with_inpersistent_state(test_case):
        for placement in all_placement():
            # logical_slice only support 1d sbp
            if len(placement.ranks.shape) != 1:
                continue
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_stateful_kernel_with_inpersistent_state(
                    test_case, placement, sbp
                )


if __name__ == "__main__":
    unittest.main()
