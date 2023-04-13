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

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


@autotest(n=3, check_graph=True)
def _test_quantile(test_cast, q):
    device = random_device()
    a = random_tensor(2, random(2, 5), random(2, 5)).to(device)
    out = torch.quantile(a, q, dim=1, interpolation="linear")
    return out


@unittest.skipIf(True, "pytorch-1.10.0 will cause oneflow cudnn or cublas error")
@flow.unittest.skip_unless_1n1d()
class TestQuantile(flow.unittest.TestCase):
    def test_quantile(test_case):
        arg_dict = OrderedDict()
        arg_dict["q"] = [0.2, 0.6, 0.8]
        for arg in GenArgList(arg_dict):
            _test_quantile(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
