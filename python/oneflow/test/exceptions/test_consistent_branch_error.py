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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n2d()
class TestConsistentBranchError(flow.unittest.TestCase):
    @autotest(n=1, check_graph=False)
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_add_with_alpha(test_case):
        try:
            flow._oneflow_internal.global_view.set_sync_timeout(10)
            data = flow.rand(2, dtype=flow.float32)
            placement = flow.env.all_device_placement("cuda")
            sbp = flow.sbp.split(0)
            global_data = data.to_global(placement=placement, sbp=sbp)

            if flow.env.get_rank() == 0:
                print(data.mean())
                print(global_data.mean())

        except Exception as e:
            err_msg = "Maybe executing different code in different ranks, please check if the code is branched and operates on the global tensor"
            assert err_msg in str(e)


if __name__ == "__main__":
    unittest.main()
