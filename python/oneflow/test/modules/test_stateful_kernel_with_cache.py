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
import os

import numpy as np
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestStatefulKernelWithInpersistentState(flow.unittest.TestCase):
    def test_stateful_kernel_with_inpersistent_state(test_case):
        x = flow.arange(4).reshape(2, 2)
        x = x.to_global(flow.placement.all("cuda"), flow.sbp.split(0))
        y = x[0:3, 0:1]
        y_np = np.array([[0], [2], [0]])
        test_case.assertTrue(
            np.array_equal(y.to_global(sbp=flow.sbp.broadcast).to_local().numpy(), y_np)
        )
        x = x.to_global(sbp=flow.sbp.split(1))
        y = x[0:3, 0:1]
        test_case.assertTrue(
            np.array_equal(y.to_global(sbp=flow.sbp.broadcast).to_local().numpy(), y_np)
        )


if __name__ == "__main__":
    unittest.main()
