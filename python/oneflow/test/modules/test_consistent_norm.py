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

@autotest(n=5, check_graph=False)
def _test_one_dim_norm_with_random_data(test_case, placement, sbp):
    input = random_tensor(ndim=4,dim0=8,dim1=8).to_consistent(
        placement=placement, sbp=sbp
    )   
    dim = random(low=0, high=4).to(int)
    ord = random().to(float)
    keepdim = random_bool()
    m = torch.linalg.norm(input, ord, dim, keepdim)
    return m

@autotest(n=5, check_graph=False)
def _test_tuple_dim_norm_with_random_data(test_case, placement, sbp):
        input = random_tensor(ndim=2,dim0=8,dim1=8)
        input = input.to_consistent(placement=placement, sbp=sbp)   
        k = random(low=-2, high=1).to(int)
        dim = oneof((-2, -1), (0, 1), (-1, 0))
        ord = oneof(float("inf"), float("-inf"), "fro", 1, -1, None)
        keepdim = random().to(bool)
        m = torch.linalg.norm(input, ord=ord, dim=dim, keepdim=keepdim)
        return m

class TestNormModule(flow.unittest.TestCase):
    @consistent
    def test_one_dim_norm_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_one_dim_norm_with_random_data(test_case, placement, sbp)
    
    @consistent
    def test_tuple_dim_norm_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_tuple_dim_norm_with_random_data(test_case, placement, sbp)

if __name__ == "__main__":
    unittest.main()

#没有问题