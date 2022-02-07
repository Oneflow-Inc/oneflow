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


@autotest(check_graph=False)
def _test_mean(test_case, placement, sbp):
    dim = random(1, 4).to(int).value()
    dim0 = random().to(int).value() * 8
    dim1 = random().to(int).value() * 8
    x = random_tensor(ndim=4, dim0=dim0, dim1=dim1, dtype=float).to_consistent(placement, sbp)
    return torch.mean(x, dim)


class TestMean(flow.unittest.TestCase):
    @consistent
    def test_mean(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_mean(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
