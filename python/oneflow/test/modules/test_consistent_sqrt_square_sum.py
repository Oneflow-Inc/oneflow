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


# raise pytorch error if open auto_backward: PyTorch error:
# element 0 of tensors does not require grad and does not have a grad_fn
@autotest(n=1, auto_backward=True, check_graph=False, rtol=0.5, atol=0.5)
def _test_sqrt_sum_with_cpu_random_data(test_case, placement, sbp):
    x = random_tensor(
        ndim=4, dim0=8, dim1=32, dim2=40, dim3=64, requires_grad=False
    ).to_global(placement=placement, sbp=sbp)
    y = torch.linalg.norm(x)
    return y


@autotest(n=1, auto_backward=False, check_graph=False, rtol=0.5, atol=0.5)
def _test_scalar_print_random_data(test_case, placement, sbp):
    x = random_tensor(
        ndim=4, dim0=8, dim1=24, dim2=16, dim3=40, requires_grad=False
    ).to_consistent(placement=placement, sbp=sbp)
    y = torch.linalg.norm(x)
    print(f"grad_norm {y.oneflow:.4f}\t")
    return y


class TestConsistentLinalgVectorNorm2D(flow.unittest.TestCase):
    @global_view
    def test_sqrt_sum_with_cpu_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_sqrt_sum_with_cpu_random_data(test_case, placement, sbp)

    @global_view
    def test_scalar_print_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_scalar_print_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
