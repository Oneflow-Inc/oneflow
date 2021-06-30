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

import oneflow.experimental as flow
from test_util import GenArgList


def _test_type_as(test_case, shape, device, src_dtype, tgt_dtype):
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=src_dtype, device=device)
    target = flow.tensor(np_input, dtype=tgt_dtype, device=device)
    input = input.type_as(target)
    test_case.assertEqual(input.dtype, target.dtype)


def _test_long(test_case, shape, device, dtype):
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=dtype, device=device)
    input = input.long()
    test_case.assertEqual(input.dtype, flow.int64)


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestTensorOps(flow.unittest.TestCase):
    def test_type_as(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2), (3, 4, 5), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["src_dtype"] = [flow.int64, flow.int32, flow.float32, flow.float64]
        arg_dict["tgt_dtype"] = [flow.int64, flow.int32, flow.float32, flow.float64]
        for arg in GenArgList(arg_dict):
            _test_type_as(test_case, *arg)

    def test_long(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2), (3, 4, 5), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [flow.int64, flow.int32, flow.float32, flow.float64]
        for arg in GenArgList(arg_dict):
            _test_long(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
