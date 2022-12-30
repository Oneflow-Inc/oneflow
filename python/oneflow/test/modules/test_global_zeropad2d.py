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

from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


@autotest(n=1, check_graph=True)
def _test_global_ZeroPad2d(test_case, placement, sbp, padding):
    x = random_tensor(ndim=4, dim0=8, dim1=16, dim2=8, dim3=8,).to_global(
        placement, sbp
    )
    m = torch.nn.ZeroPad2d(padding)
    y = m(x)
    return y


class TestGlobalZeroPad2dModule(flow.unittest.TestCase):
    @globaltest
    def test_global_ZeroPad2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["padding"] = [2, (1, 1, 2, 2)]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=4):
                    _test_global_ZeroPad2d(test_case, placement, sbp, *arg)


if __name__ == "__main__":
    unittest.main()
