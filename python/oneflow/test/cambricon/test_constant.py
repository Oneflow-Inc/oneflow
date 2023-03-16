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

import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList


def _test_different_dtype(test_case, shape, dtype):
    y1 = flow.ones(shape, device=flow.device("mlu"), dtype=dtype)
    y2 = flow.ones(shape, device="cpu", dtype=dtype)
    test_case.assertTrue(np.array_equal(y1.numpy(), y2.numpy()))
    y1 = flow.full(shape, 2.0, device=flow.device("mlu"), dtype=dtype)
    y2 = flow.full(shape, 2.0, device="cpu", dtype=dtype)
    test_case.assertTrue(np.array_equal(y1.numpy(), y2.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestCambriconConstantModule(flow.unittest.TestCase):
    def test_constant(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_different_dtype,
        ]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
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
