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
import os
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _dtype_flow_to_np(dtype):
    return {flow.float32: np.float32, flow.float16: np.float16}[dtype]


def _np_softmax(x, dtype=None):
    if dtype is not None:
        x = x.astype(dtype)
    x -= np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=-1, keepdims=True)


def _test_softmax_impl(test_case, shape, input_dtype, output_dtype):
    np_input = np.random.randn(*shape).astype(_dtype_flow_to_np(input_dtype))
    of_input = flow.tensor(np_input, dtype=input_dtype, device=flow.device("cuda"))
    of_out = flow.nn.functional.softmax(of_input, dtype=output_dtype)
    if output_dtype is not None:
        np_out = _np_softmax(np_input, dtype=_dtype_flow_to_np(output_dtype))
    else:
        np_out = _np_softmax(np_input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.001, 0.001))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class Testsoftmax(flow.unittest.TestCase):
    def test_softmax(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(20, 30), (32, 128)]
        arg_dict["input_dtype"] = [flow.float16, flow.float32]
        arg_dict["output_dtype"] = [None, flow.float32]
        for arg in GenArgList(arg_dict):
            _test_softmax_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
