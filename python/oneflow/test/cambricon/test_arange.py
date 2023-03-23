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


def _test_arange(test_case, device, dtype):
    of_cpu_out = flow.arange(
        13, device="cpu", dtype=dtype if dtype != flow.float16 else flow.float32
    )
    of_out = flow.arange(13, device=device, dtype=dtype)
    test_case.assertTrue(np.allclose(of_out.numpy(), of_cpu_out.numpy(), 1e-05, 1e-05))


def _test_arange_step_prarm(test_case, device, dtype):
    of_cpu_out = flow.arange(
        0, 20, 2, device="cpu", dtype=dtype if dtype != flow.float16 else flow.float32
    )
    of_out = flow.arange(0, 20, step=2, device=device, dtype=dtype)
    test_case.assertTrue(np.allclose(of_out.numpy(), of_cpu_out.numpy(), 1e-05, 1e-05))


def _test_arange_more_params(test_case, device, dtype):
    of_cpu_out = flow.arange(
        0, 100, 3, device="cpu", dtype=dtype if dtype != flow.float16 else flow.float32
    )
    of_out = flow.arange(start=0, end=100, step=3, device=device, dtype=dtype)
    test_case.assertTrue(np.allclose(of_out.numpy(), of_cpu_out.numpy(), 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestArange(flow.unittest.TestCase):
    def test_arange(test_case):
        arg_dict = OrderedDict()
        arg_dict["function_test"] = [
            _test_arange,
            _test_arange_step_prarm,
            _test_arange_more_params,
        ]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
            flow.int8,
            flow.int32,
            flow.int64,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
