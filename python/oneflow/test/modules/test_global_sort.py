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

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_sort_impl(test_case, placement):
    sbp = random_sbp(placement, max_dim=4)
    x_dims = [random(2, 4) * 8 for _ in range(4)]
    x = random_tensor(4, *x_dims)
    dim = random(0, 4).to(int).value()
    descending = random().to(bool).value()

    y = x.to_global(placement=placement, sbp=sbp)
    sort_result = torch.sort(y, dim=dim, descending=descending)
    value = sort_result[0]
    return value


class TestSortGlobal(flow.unittest.TestCase):
    @globaltest
    def test_sort(test_case):
        for placement in all_placement():
            _test_sort_impl(test_case, placement)


if __name__ == "__main__":
    unittest.main()
