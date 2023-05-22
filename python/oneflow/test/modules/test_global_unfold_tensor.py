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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *
import numpy as np


@autotest(n=1, auto_backward=True, check_graph=True)
def _test_global_unfold_tensor_with_random_data(test_case, placement, sbp):
    ndim = 4
    dim = random(0, ndim).to(int).value()
    x = random_tensor(
        ndim=ndim,
        dim0=random(1, 3).to(int) * 8,
        dim1=random(1, 3).to(int) * 8,
        dim2=4,
        dim3=4,
    ).to_global(placement, sbp)
    high = x.oneflow.size()[dim]
    size = random(1, high).to(int).value()
    step = random(1, high).to(int).value()
    y = x.unfold(dim, size, step)
    return y


class TestGlobalUnfoldTensor(flow.unittest.TestCase):
    @globaltest
    def test_global_unfold_tensor_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_unfold_tensor_with_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
