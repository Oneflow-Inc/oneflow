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
from oneflow.test_utils.automated_test_util import *
import oneflow.unittest
from collections import OrderedDict


def _test_finfo(test_case, dtype):
    finfo = torch.finfo(dtype)
    torch_finfo = finfo.pytorch
    flow_finfo = finfo.oneflow
    test_case.assertEqual(torch_finfo.max, flow_finfo.max)
    test_case.assertEqual(torch_finfo.min, flow_finfo.min)
    test_case.assertEqual(torch_finfo.bits, flow_finfo.bits)
    test_case.assertEqual(torch_finfo.eps, flow_finfo.eps)
    test_case.assertEqual(torch_finfo.tiny, flow_finfo.tiny)
    test_case.assertEqual(torch_finfo.resolution, flow_finfo.resolution)


@flow.unittest.skip_unless_1n1d()
class TestIInfo(flow.unittest.TestCase):
    @autotest(n=3, check_graph=False)
    def test_iinfo_max(test_case):
        for dtype in [torch.uint8, torch.int8, torch.int32, torch.int64]:
            return torch.iinfo(dtype).max

    @autotest(n=3, check_graph=False)
    def test_iinfo_min(test_case):
        for dtype in [torch.uint8, torch.int8, torch.int32, torch.int64]:
            return torch.iinfo(dtype).min

    @autotest(n=3, check_graph=False)
    def test_iinfo_bits(test_case):
        for dtype in [torch.uint8, torch.int8, torch.int32, torch.int64]:
            return torch.iinfo(dtype).bits

    @autotest(n=3, check_graph=False)
    def test_finfo_min(test_case):
        for dtype in [torch.float16, torch.float32, torch.float64]:
            _test_finfo(test_case, dtype)


if __name__ == "__main__":
    unittest.main()
