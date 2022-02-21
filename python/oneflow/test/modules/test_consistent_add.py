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
def _test_add_with_alpha(test_case, placement, sbp):
    x1 = random_tensor(2, 8, 8).to_global(placement=placement, sbp=sbp).mean()
    x2 = random_tensor(2, 8, 8).to_global(placement=placement, sbp=sbp).mean()
    x3 = random_tensor(2, 8, 8).to_global(placement=placement, sbp=sbp).mean()
    y = random_tensor(2, 8, 8).to_global(placement=placement, sbp=sbp)
    s = random().to(float)
    alpha = random().to(float)
    z1 = torch.add(x1, y, alpha=alpha)
    z2 = torch.add(x2, s, alpha=alpha)
    z3 = torch.add(s, x3, alpha=alpha)
    return z1, z2, z3


class TestAddModule(flow.unittest.TestCase):
    @globaltest
    def test_add_with_alpha(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_add_with_alpha(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
