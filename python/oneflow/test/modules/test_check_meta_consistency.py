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

import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList


@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGlobalCastModule_1n2d(flow.unittest.TestCase):
    def test_check_meta_consistency(test_case):
        if os.getenv("RANK") == "0":
            x = flow.ones((16, 16), device=flow.device("cuda"), dtype=flow.int32)
        else:
            x = flow.zeros((1,), device=flow.device("cuda"), dtype=flow.float)
        placement = flow.placement("cuda", ranks=[0])
        sbp = (flow.sbp.broadcast,)
        y = x.to_global(placement=placement, sbp=sbp)
        y.check_meta_consistency()
        y = y.to_global(sbp=flow.sbp.split(0))
        y.check_meta_consistency()


if __name__ == "__main__":
    unittest.main()
