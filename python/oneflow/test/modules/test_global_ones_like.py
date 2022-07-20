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

from oneflow.test_utils.automated_test_util import *


def _test_ones_like_float(test_case, placement, sbp, shape, device):
    x = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    x = x.to_global(placement=placement, sbp=sbp)
    y = flow.ones_like(x, placement=placement, sbp=sbp)
    test_case.assertTrue(y.dtype is flow.float32)
    test_case.assertTrue(y.shape == x.shape)
    test_case.assertTrue(y.placement == placement)
    y_numpy = np.ones(x.numpy().shape)
    print("y_numpy: ", y_numpy)
    print("y.numpy()", y.numpy())

    test_case.assertTrue(np.array_equal(y.numpy(), y_numpy))


def _test_ones_like_int(test_case, placement, sbp, shape, device):
    x = flow.tensor(np.random.randn(*shape), dtype=flow.int, device=flow.device(device))
    x = x.to_global(placement=placement, sbp=sbp)
    y = flow.ones_like(x, dtype=flow.int, placement=placement, sbp=sbp)
    test_case.assertTrue(y.dtype is flow.int)
    test_case.assertTrue(y.shape == x.shape)
    test_case.assertTrue(y.placement == placement)
    y_numpy = np.ones(x.numpy().shape)
    test_case.assertTrue(np.array_equal(y.numpy(), y_numpy))


class TestModule(flow.unittest.TestCase):
    @unittest.skip("TODO: global ones_like test will fail!")
    @globaltest
    def test_ones_like(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_ones_like_float, _test_ones_like_int]
        arg_dict["shape"] = [(8, 8), (8, 8, 4), (8, 8, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=2):
                    arg[0](test_case, placement, sbp, *arg[1:])


if __name__ == "__main__":
    unittest.main()
