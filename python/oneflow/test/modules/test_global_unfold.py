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
from oneflow.nn.common_types import _size_2_t


@autotest(n=1, check_graph=True)
def _test_unfold_with_random_data(test_case, placement, sbp):
    ndim = 4
    dims = [random(1, 3).to(int).value() * 8 for i in range(ndim)]
    m = torch.nn.Unfold(
        kernel_size=random(1, 3).to(_size_2_t),
        dilation=random(1, 2).to(_size_2_t),
        padding=random(0, 1).to(_size_2_t),
        stride=random(1, 2).to(_size_2_t),
    )
    m.train(random())

    x = random_tensor(ndim, *dims).to_global(placement, sbp)
    y = m(x)
    func_y = torch.nn.functional.unfold(
        x,
        kernel_size=random(1, 3).to(_size_2_t),
        dilation=random(1, 2).to(_size_2_t),
        padding=random(0, 1).to(_size_2_t),
        stride=random(1, 2).to(_size_2_t),
    )
    return y, func_y


class TestUnfold(flow.unittest.TestCase):
    @globaltest
    def test_unfold_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_unfold_with_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
