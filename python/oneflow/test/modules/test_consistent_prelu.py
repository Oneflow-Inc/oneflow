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

from oneflow.test_utils.automated_test_util import *
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

@autotest(n=5, check_graph=False)
def _test_prelu_4dim_module_with_random_data(test_case, placement, sbp):
    x = random_tensor(ndim=4, dim0=8 ,dim1=8)
    x = x.to_consistent(placement=placement, sbp=sbp)
    m = torch.nn.PReLU(
        num_parameters=8 | nothing(), init=random().to(float) | nothing(),
    )
    weight_sbp = random_sbp(placement, max_dim=1)
    m.weight.to_consistent(placement=placement, sbp=weight_sbp)
    m.train(random())
    y = m(x)
    return y

class TestModule(flow.unittest.TestCase):
    @consistent
    def test_prelu_4dim_module_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_prelu_4dim_module_with_random_data(test_case, placement, sbp)

if __name__ == "__main__":
    unittest.main()

#2卡有问题