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

from random import choice
import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *

@autotest(check_graph=False)
def _test_clamp(test_case, placement, sbp):
    x = random_pytorch_tensor(
        ndim=4,
        dim1=random(low=4, high=8).to(int),
        dim2=random(low=4, high=8).to(int),
        dim3=random(low=4, high=8).to(int),
    ).to_consistent(
        placement=placement, sbp=sbp
    )
    arg = choice([(None, 0.5), (0.1, None), (0.1, 0.5)])
    y = torch.clamp(x, arg[0], arg[1])
    return y


class TestModule(flow.unittest.TestCase):
    @consistent
    def test_bmm_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_clamp(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
