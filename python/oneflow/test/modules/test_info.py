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


def _test_finfo(test_case, dtype):
    # test finfo without input params
    if dtype is None:
        finfo = torch.finfo()
    else:
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
class TestTypeInfo(flow.unittest.TestCase):
    def test_iinfo(test_case):
        for dtype in [torch.uint8, torch.int8, torch.int32, torch.int64]:
            iinfo = torch.iinfo(dtype)
            # checker not implemented for type <class 'torch.iinfo'> and <class 'oneflow.iinfo'>
            # so return all fields as a tuple
            return iinfo.max, iinfo.min, iinfo.bits

    def test_finfo(test_case):
        for dtype in [None, torch.half, torch.bfloat16, torch.float, torch.double]:
            _test_finfo(test_case, dtype)


if __name__ == "__main__":
    unittest.main()
