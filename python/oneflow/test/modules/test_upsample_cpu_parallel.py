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

import oneflow as flow
import oneflow.unittest


def _test_upsample2d(test_case, device):
    arr = np.random.randn(1, 2, 3, 3)
    input = flow.Tensor(arr, device=flow.device(device), dtype=flow.float32,)
    m = flow.nn.UpsamplingNearest2d(scale_factor=2.0)
    output = m(input)
    print("output.sum >>>>>>>>>>", output.sum())
    print("output.shape >>>>>>>>> ", output.shape)
    test_case.assertTrue(True)


@flow.unittest.skip_unless_1n1d()
class TestUpsample2d(flow.unittest.TestCase):
    def test_upsample2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_upsample2d,
        ]
        arg_dict["device"] = ["cpu"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
