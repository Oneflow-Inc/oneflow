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
import random
import numpy as np
from collections import OrderedDict

import oneflow as flow

import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList


def _test_is_view(test_case, device):
    shape = (2, 3, 4, 5)
    xx = flow.randn(shape, device=device)
    yy = xx.reshape(4, 5, 6)
    test_case.assertEqual(xx.is_contiguous(), yy.is_contiguous())
    test_case.assertEqual(yy.is_view(), True)
    test_case.assertEqual(xx.is_view(), False)


@flow.unittest.skip_unless_1n1d()
class TestTensorIsView(flow.unittest.TestCase):
    def test_is_view(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        for arg in GenArgList(arg_dict):
            _test_is_view(test_case, *arg[0:])


if __name__ == "__main__":
    unittest.main()
