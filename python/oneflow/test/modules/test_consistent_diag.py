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
def test_diag_impl(test_case, ndim, placement, sbp):
    dim0 = np.random.randint(4, 10)
    dim1 = np.random.randint(4, 10)
    if ndim==1:
        x = random_pytorch_tensor(1, dim0)
    elif ndim==2:
        x = random_pytorch_tensor(2, dim0, dim1)
    y = x.to_consistent(placement=placement, sbp=sbp)
    return torch.diag(y)

@flow.unittest.skip_unless_1n1d()
class TestDiagConsistent(flow.unittest.TestCase):
    @consistent
    def test_diag(test_case):
        # random ndim in range [1,2]
        ndim = np.random.randint(1, 3)
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                test_diag_impl(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()
