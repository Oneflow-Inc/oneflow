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

from test_util import GenArgList
import oneflow.experimental as flow

def _test_dot(test_case, device, dtype):
    np_x = np.random.randn(1000).astype(dtype)
    np_y = np.random.randn(1000).astype(dtype)

    np_out = np.dot(np_x, np_y)

    x = flow.tensor(np_x, device=flow.device(device))
    y = flow.tensor(np_y, device=flow.device(device))

    out = flow.dot(x, y)

    test_case.assertTrue(np.allclose(np_out, out.numpy(), rtol=1e-04, atol=1e-10))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestDot(flow.unittest.TestCase):

    def test_cpu_dot(test_case):
        arg_dict = OrderedDict()
        arg_dict["dtype"] = [np.int32, np.int64, np.float32, np.double]
        for arg in GenArgList(arg_dict):
            _test_dot(test_case, "cpu", *arg)

    def test_gpu_dot(test_case):
        arg_dict = OrderedDict()
        arg_dict["dtype"] = [np.float32, np.double]
        for arg in GenArgList(arg_dict):
            _test_dot(test_case, "cuda", *arg)


if __name__ == "__main__":
    unittest.main()
