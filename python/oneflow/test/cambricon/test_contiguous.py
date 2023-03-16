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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_contiguous_forward(test_case, device, dtype):
    shape = [2, 3]
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device), dtype=dtype)
    y = x.permute(1, 0)
    test_case.assertTrue(not y.is_contiguous())
    z = y.contiguous()
    test_case.assertTrue(z.is_contiguous())
    z_cpu = x.cpu().permute(1, 0).contiguous()
    test_case.assertTrue(np.allclose(z.cpu().numpy(), z_cpu.numpy()))

    shape = [2, 3, 4]
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device), dtype=dtype)
    y = x.permute(1, 2, 0)
    test_case.assertTrue(not y.is_contiguous())
    z = y.contiguous()
    test_case.assertTrue(z.is_contiguous())
    z_cpu = x.cpu().permute(1, 2, 0).contiguous()
    test_case.assertTrue(np.allclose(z.cpu().numpy(), z_cpu.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestContiguousCambriconModule(flow.unittest.TestCase):
    def test_contiguous(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_contiguous_forward,
        ]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
            flow.int8,
            flow.uint8,
            flow.int32,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
