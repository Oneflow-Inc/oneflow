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


@autotest(n=10, check_graph=False)
def _test_upsample(test_case, ndim, placement, sbp):

    dim = [0]*5
    dim[0] = np.random.randint(1, 2) * 8
    dim[1] = np.random.randint(1, 2) * 8
    dim[2] = np.random.randint(1, 2) * 8
    dim[3] = np.random.randint(1, 2) * 8
    dim[4] = np.random.randint(1, 2) * 8
    
   
    x = random_pytorch_tensor(4, dim[0],dim[1],dim[2],dim[3]).to_consistent(placement=placement, sbp=sbp)   
  
    
   
    m = flow.nn.Upsample(scale_factor=2.0, mode="nearest")
    z = m(x)
    return z


class TestUpsampleConsistent(flow.unittest.TestCase):
    @consistent
    def test_upsample(test_case):
        ndim = np.random.randint(1, 6)
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                _test_upsample(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()
