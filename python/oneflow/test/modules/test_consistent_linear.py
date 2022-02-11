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

from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=False)
def _test_linear_with_random_data(test_case, placement, weight_sbp, input_sbp):
    print(placement)
    print(weight_sbp)
    input_size = 8
    m = torch.nn.Linear(in_features=input_size, out_features=8, bias=random())
    m.train(random())
    m.weight = torch.nn.Parameter(
        m.weight.to_global(placement=placement, sbp=weight_sbp)
    )
    if m.bias is not None:
        # bias is 1-d tensor
        bias_sbp = random_sbp(placement, max_dim=1)
        m.bias = torch.nn.Parameter(m.bias.to_global(placement=placement, sbp=bias_sbp))
    x = random_tensor(ndim=2, dim1=input_size, dim2=8).to_global(
        placement=placement, sbp=input_sbp
    )
    y = m(x)
    return y


# class TestLinearModule(flow.unittest.TestCase):
#     @global_view
#     def test_linear_with_random_data(test_case):
#         for placement in all_placement():
#             # TODO(): Fix 2d sbp
#             if len(placement.hierarchy) != 1:
#                 continue
#             for sbp in all_sbp(placement, max_dim=2):
#                 _test_linear_with_random_data(test_case, placement, sbp, sbp)


if __name__ == "__main__":
    unittest.main()
