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

import numpy as np
import oneflow as flow
import os

from oneflow.test_utils.automated_test_util.torch_flow_dual_object import globaltest
from oneflow.test_utils.test_util import GenArgList


def _test_auto_to_global(test_case, device):
    os.environ["ONEFLOW_ENABLE_GLOBAL_INPUTS_WITH_INCONSISTENT_PLACEMENT"] = "true"
    x = flow.ones(
        (2, 2),
        sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
        placement=flow.placement(device, ranks=[[0], [1]]),
    )
    y = flow.zeros(
        (2, 2),
        sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
        placement=flow.placement(device, ranks=[[2], [3]]),
    )
    z = x + y
    test_case.assertTrue(np.array_equal(x.numpy(), z.numpy()))
    test_case.assertEqual(y.placement, z.placement)
    os.environ["ONEFLOW_ENABLE_GLOBAL_INPUTS_WITH_INCONSISTENT_PLACEMENT"] = "false"


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestAutoToGlobal(flow.unittest.TestCase):
    @globaltest
    @flow.unittest.skip_unless_1n4d()
    def test_auto_to_global(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        for arg in GenArgList(arg_dict):
            _test_auto_to_global(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
