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
from automated_test_util import *


def _test_0_dim_tensor(test_case, device):
    scalar = 9.999
    input_np = np.array(scalar)
    input = flow.Tensor(input_np)
    # print("input >>>>>>>>>>>>>> ", input) 
    # # input >>>>>>>>>>>>>>  tensor(9.999, dtype=oneflow.float32)
    test_case.assertEqual(input.numel(), 1)
    test_case.assertEqual(input.ndimension(), 0)


@flow.unittest.skip_unless_1n1d()
class TestZeroDimensionTensor(flow.unittest.TestCase):
    def test_0_dim_tensor(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_0_dim_tensor,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
