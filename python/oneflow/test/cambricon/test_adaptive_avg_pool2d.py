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


def _test_adaptive_avg_pool2d_forward(test_case, shape, out_shape, device, dtype):
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device), dtype=dtype)
    pool = flow.nn.AdaptiveAvgPool2d((out_shape[2], out_shape[3]))
    y = pool(x)
    y_cpu = pool(x.to("cpu"))
    test_case.assertTrue(np.allclose(y.numpy(), y_cpu.numpy(), 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestAdaptiveAvgPool2dCambriconModule(flow.unittest.TestCase):
    def test_add(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_adaptive_avg_pool2d_forward,
        ]
        arg_dict["shape"] = [(1, 2, 224, 224), (1, 3, 128, 128)]
        arg_dict["out_shape"] = [(1, 2, 64, 64), (1, 3, 32, 35)]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
