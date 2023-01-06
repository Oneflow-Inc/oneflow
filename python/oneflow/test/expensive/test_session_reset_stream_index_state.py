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

from random import shuffle
import numpy as np
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest
import oneflow.framework.session_context as session_ctx


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestSessionResetStreamIndexStateMock1(flow.unittest.TestCase):
    @autotest(n=1000, auto_backward=False)
    def test_mock_1(test_case):
        device = gpu_device()
        x = random_tensor().to(device)
        return torch.nn.functional.relu(x)

    def setUp(self):
        session_ctx.GetDefaultSession().Reset()


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestSessionResetStreamIndexStateMock2(flow.unittest.TestCase):
    @autotest(n=1000, auto_backward=False)
    def test_mock_2(test_case):
        device = gpu_device()
        input_size = random()
        m = torch.nn.Linear(in_features=input_size, out_features=random())
        m.to(device)
        x = random_tensor(ndim=2, dim1=input_size).to(device)
        y = m(x)
        return y

    def setUp(self):
        # If not reset session, will raise
        # F20221201 13:55:30.758263 2023961 stream_id.h:33] Check failed: stream_index <= kMaxStreamIndex (4096 vs. 4095)
        session_ctx.GetDefaultSession().Reset()


if __name__ == "__main__":
    unittest.main()
