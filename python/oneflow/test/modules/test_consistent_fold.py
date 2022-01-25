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

@autotest(n=3, auto_backward=True, rtol=1e-4, atol=1e-4)
def test_fold_impl(test_case, placement, sbp):
    m = torch.nn.Fold(
        output_size=constant((4, 4)),
        kernel_size=constant(3),
        dilation=constant(1),
        padding=constant(1),
        stride=constant(1),
    )
    m.train(random())
    device = random_device()
    m.to(device)
    x = random_pytorch_tensor(
        ndim=3, dim0=constant(2), dim1=constant(36), dim2=constant(16)
    ).to_consistent(placement=placement, sbp=sbp)
    y = m(x)
    return y

class TestFold(flow.unittest.TestCase):
    @consistent
    def test_fold(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                test_fold_impl(test_case, placement, sbp)

if __name__ == "__main__":
    unittest.main()
