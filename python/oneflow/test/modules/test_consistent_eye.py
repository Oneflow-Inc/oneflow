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


@autotest(n=10, auto_backward=False, check_graph=False)
def test_eye_impl(test_case, placement, sbp):
    n = np.random.randint(1, 10) * 8
    m = np.random.randint(1, 10) * 8
    x = torch.eye(n, m)
    y = x.to_consistent(placement=placement, sbp=sbp)
    return y

class TestEyeConsistent(flow.unittest.TestCase):
    @consistent
    def test_eye(test_case):
        shape = random_tensor().value().shape
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                test_eye_impl(test_case, placement, sbp)

if __name__ == "__main__":
    unittest.main()
